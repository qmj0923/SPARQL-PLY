'''
Copyright (c) 2024 qmj0923
https://github.com/qmj0923/SPARQL-PLY
'''

from __future__ import annotations
from collections import defaultdict
from collections.abc import Sequence
from typing import (
    List, Tuple, Set, Dict, Callable, Optional, Union, Type,
)

from sparql_ply.components import (
    NodeTerm, PropertyPath, CollectionPath, BlankNodePath, TriplesPath,
    Expression, GraphPattern, Query, ComponentWrapper, QueryComponent,
)
from sparql_ply.nested_replacer import (
    ReplacerNode, ReplacerFactory,
)
from sparql_ply.sparql_yacc import parse as parse_sparql


def traverse(
    component: QueryComponent,
    before: Callable[[QueryComponent], None] = lambda x: None,
    after: Callable[[QueryComponent], None] = lambda x: None,
    skip: Callable[[QueryComponent], bool] = lambda x: False,
    prune: Callable[[QueryComponent], bool] = lambda x: False,
):
    if skip(component):
        return
    before(component)
    if not prune(component):
        if isinstance(component, NodeTerm):
            pass
        elif isinstance(component, (
            PropertyPath, CollectionPath, Expression, GraphPattern
        )):
            for child in component.children:
                traverse(child, before, after)
        elif isinstance(component, BlankNodePath):
            for p, ol in component.pred_obj_list:
                traverse(p, before, after)
                for o in ol:
                    traverse(o, before, after)
        elif isinstance(component, TriplesPath):
            traverse(component.subj, before, after)
            if component.pred_obj_list is not None:
                for p, ol in component.pred_obj_list:
                    traverse(p, before, after)
                    for o in ol:
                        traverse(o, before, after)
        elif isinstance(component, Query):
            if component.target is not None:
                if isinstance(component.target, (NodeTerm, GraphPattern)):
                    traverse(component.target, before, after)
                elif isinstance(component.target, Sequence):
                    for t in component.target:
                        if isinstance(t, NodeTerm):
                            traverse(component.target, before, after)
                        elif isinstance(t, ComponentWrapper):
                            traverse(t[0], before, after)
                            traverse(t[1], before, after)
                        else:
                            raise NotImplementedError
                else:
                    raise NotImplementedError
            if component.dataset is not None:
                for d in component.dataset:
                    traverse(d[-1], before, after)
            if component.pattern is not None:
                traverse(component.pattern, before, after)
            if component.group_by is not None:
                for x in component.group_by:
                    if isinstance(x, ComponentWrapper):
                        traverse(x[0], before, after)
                        traverse(x[1], before, after)
                    elif isinstance(x, (NodeTerm, Expression)):
                        traverse(x, before, after)
                    else:
                        raise NotImplementedError
            if component.having is not None:
                for x in component.having:
                    traverse(x, before, after)
            if component.order_by is not None:
                for x in component.order_by:
                    if isinstance(x, ComponentWrapper):
                        traverse(x[1], before, after)
                    elif isinstance(x, (NodeTerm, Expression)):
                        traverse(x, before, after)
                    else:
                        raise NotImplementedError
            if component.limit is not None:
                traverse(component.limit, before, after)
            if component.offset is not None:
                traverse(component.offset, before, after)
            if component.values is not None:
                for row in component.values:
                    for x in row:
                        traverse(x, before, after)
    after(component)


def collect_component(
    component: QueryComponent,
    cls: Union[Type[QueryComponent], Tuple[Type[QueryComponent], ...]],
    typ: Optional[int] = None,
    skip: Callable[[QueryComponent], bool] = lambda x: False,
    prune: Callable[[QueryComponent], bool] = lambda x: False,
) -> List[QueryComponent]:
    def func(x: QueryComponent):
        nonlocal res
        if isinstance(x, cls) and (typ is None or x.type & typ):
            res.append(x)

    res = []
    traverse(component, func, skip=skip, prune=prune)
    return res


class LabelGenerator:
    def __init__(
        self,
        gen_func: Callable[[int], str] = lambda x: str(x),
        skip_set: Optional[Set[str]] = None,
        start: int = 0,
    ):
        self.gen_func = gen_func
        self.skip_set = skip_set if skip_set is not None else set()
        self.counter = start
    
    def __call__(self) -> str:
        for _ in range(len(self.skip_set) + 1):
            res = self.gen_func(self.counter)
            self.counter += 1
            if res not in self.skip_set:
                break
        return res


class SyntaxFormExpander:
    '''
    Expand abbreviations for IRIs and triple patterns

    https://www.w3.org/TR/2013/REC-sparql11-query-20130321/#sparqlExpandForms
    '''

    KEY_A = 'a'
    KEY_PN = 'NodeTerm' + str(NodeTerm.PREFIXED_NAME)
    KEY_IRIREF = 'NodeTerm' + str(NodeTerm.IRIREF)

    def __init__(self, prefix_dict: Optional[dict[str, str]] = None):
        self.key2spans: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.prefix_dict: Dict[str, str] = (
            prefix_dict if prefix_dict is not None else dict()
        )
        self.base: Optional[str] = None
        self.blank_gen: Optional[LabelGenerator] = None

    #################### expand node term ####################
    
    def _collect_prologue(self, query: Query):
        if query.prologue is None:
            return
        for row in reversed(query.prologue):
            if len(row) == 2:
                self.base = str(row[1])
            elif len(row) == 3:
                prefix = str(row[1])[:-1]
                self.prefix_dict[prefix] = str(row[2])
            else:
                raise NotImplementedError

    def _record_node_span(self, x: QueryComponent):
        if not isinstance(x, NodeTerm):
            return
        if x.type & NodeTerm.SPECIAL and x.value == 'a':
            self.key2spans[self.KEY_A].append([x.lexstart, x.lexstop])
        elif x.type & NodeTerm.PREFIXED_NAME:
            self.key2spans[self.KEY_PN].append([x.lexstart, x.lexstop])
        elif x.type & NodeTerm.IRIREF and self.base is not None:
            self.key2spans[self.KEY_IRIREF].append([x.lexstart, x.lexstop])

    def _prefixed_name_handler(self, node: ReplacerNode) -> ReplacerNode:
        res = pn = node.remainder[0]
        prefix, val = pn.split(':', 1)
        if prefix in self.prefix_dict:
            mapped_iri = self.prefix_dict[prefix][1:-1]
            idx = mapped_iri.rfind('/')
            mapped_iri = '/' if idx == -1 else mapped_iri[:idx + 1]
            res = '<' + mapped_iri + val + '>'
        else:
            raise ValueError(f'Prefix {prefix} not found')
        return ReplacerFactory.create_leaf(res, None)

    def _iriref_handler(self, node: ReplacerNode) -> ReplacerNode:
        res = iriref = node.remainder[0]
        if self.base is not None:
            mapped_iri = self.base[1:-1]
            idx = mapped_iri.rfind('/')
            mapped_iri = '/' if idx == -1 else mapped_iri[:idx + 1]
            res = '<' + mapped_iri + iriref[1:-1] + '>'
        return ReplacerFactory.create_leaf(res, None)

    @classmethod
    def expand_node_term(
        cls, sparql: str,
        default_prefix_dict: Optional[dict[str, str]] = None,
    ) -> str:
        '''
        Expand abbreviations for IRIs.
        '''
        query = parse_sparql(sparql)
        this = cls(default_prefix_dict)
        this._collect_prologue(query)
        traverse(query, this._record_node_span)
        rt = ReplacerFactory.create_tree(sparql, this.key2spans)
        handler_dict = {
            cls.KEY_A: '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>',
            cls.KEY_PN: this._prefixed_name_handler,
            cls.KEY_IRIREF: this._iriref_handler,
        }
        return rt.replace(handler_dict).to_str()

    #################### expand triples path ####################
    
    def _process_object_node(
        self, subj: NodeTerm, pred: Union[NodeTerm, PropertyPath],
        obj_node: Union[NodeTerm, CollectionPath, BlankNodePath],
    ) -> List[Tuple[
        NodeTerm, Union[NodeTerm, PropertyPath], NodeTerm,
    ]]:
        if isinstance(obj_node, NodeTerm):
            triples = [[subj, pred, obj_node]]
        elif isinstance(obj_node, CollectionPath):
            obj, nt_list = self._process_rdf_collection(obj_node)
            triples = [[subj, pred, obj]] + nt_list
        elif isinstance(obj_node, BlankNodePath):
            obj, nt_list = self._process_blank_node(obj_node)
            triples = [[subj, pred, obj]] + nt_list
        else:
            raise NotImplementedError
        return triples

    def _process_rdf_collection(
        self, collection: CollectionPath,
    ) -> Tuple[NodeTerm, List[Tuple[
        NodeTerm, Union[NodeTerm, PropertyPath], NodeTerm,
    ]]]:
        RDF_FIRST = '<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>'
        RDF_REST = '<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>'
        RDF_NIL = '<http://www.w3.org/1999/02/22-rdf-syntax-ns#nil>'
        nt_first = NodeTerm(-1, -1, RDF_FIRST, NodeTerm.IRIREF)
        nt_rest = NodeTerm(-1, -1, RDF_REST, NodeTerm.IRIREF)
        nt_nil = NodeTerm(-1, -1, RDF_NIL, NodeTerm.IRIREF)
        
        subj = NodeTerm(-1, -1, self.blank_gen(), NodeTerm.BLANK_NODE_LABEL)
        triples = list()
        last = subj
        for i, x in enumerate(collection.children):
            triples += self._process_object_node(last, nt_first, x)

            if i == len(collection.children) - 1:
                triples.append([last, nt_rest, nt_nil])
                break
            else:
                obj = NodeTerm(
                    -1, -1, self.blank_gen(),
                    NodeTerm.BLANK_NODE_LABEL
                )
                triples.append([last, nt_rest, obj])
                last = obj
        return subj, triples

    def _process_blank_node(
        self, blank_node: BlankNodePath,
    ) -> Tuple[NodeTerm, List[Tuple[
        NodeTerm, Union[NodeTerm, PropertyPath], NodeTerm,
    ]]]:
        subj = NodeTerm(-1, -1, self.blank_gen(), NodeTerm.BLANK_NODE_LABEL)
        triples = list()
        for pred, obj_list in blank_node.pred_obj_list:
            for obj in obj_list:
                triples += self._process_object_node(subj, pred, obj)
        return subj, triples

    def _process_triples_path(
        self, triples_path: TriplesPath
    ) -> List[Tuple[
        NodeTerm, Union[NodeTerm, PropertyPath], NodeTerm,
    ]]:
        if isinstance(triples_path.subj, NodeTerm):
            subj, triples = triples_path.subj, list()
        elif isinstance(triples_path.subj, CollectionPath):
            subj, triples = self._process_rdf_collection(triples_path.subj)
        elif isinstance(triples_path.subj, BlankNodePath):
            subj, triples = self._process_blank_node(triples_path.subj)
        else:
            raise NotImplementedError
        
        if triples_path.pred_obj_list is None:
            return triples
        for pred, obj_list in triples_path.pred_obj_list:
            for obj in obj_list:
                triples += self._process_object_node(subj, pred, obj)
        return triples

    @classmethod
    def expand_triples_path(cls, sparql: str) -> str:
        '''
        Expand abbreviations for triple patterns.
        '''
        query = parse_sparql(sparql)
        this = cls()
        blank_labels = collect_component(
            query, NodeTerm, NodeTerm.BLANK_NODE_LABEL
        )
        this.blank_gen = LabelGenerator(
            lambda x: f'_:b{x}',
            {str(x) for x in blank_labels},
        )

        tp_list = collect_component(
            query, TriplesPath, 
            prune=lambda x: isinstance(x, TriplesPath),
        )
        expanded_list = list()
        for tp in tp_list:
            triples = this._process_triples_path(tp)
            new_str = '.\n'.join(
                ' '.join(str(x) for x in triple)
                for triple in triples
            )
            expanded_list.append(new_str)

        tp_spans = [[tp.lexstart, tp.lexstop] for tp in tp_list]
        key2spans = {str(i): [span] for i, span in enumerate(tp_spans)}
        handler_dict = {str(i): x for i, x in enumerate(expanded_list)}
        rt = ReplacerFactory.create_tree(sparql, key2spans)
        return rt.replace(handler_dict).to_str()

    ############################################################

    @classmethod
    def run(
        cls, sparql: str,
        default_prefix_dict: Optional[dict[str, str]] = None,
    ) -> str:
        '''
        Expand abbreviations for IRIs and triple patterns.
        '''
        sparql = cls.expand_node_term(sparql, default_prefix_dict)
        return cls.expand_triples_path(sparql)

