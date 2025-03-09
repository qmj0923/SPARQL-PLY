'''
Copyright (c) 2024 qmj0923
https://github.com/qmj0923/SPARQL-PLY
'''

from __future__ import annotations
import os
import sys

# https://stackoverflow.com/questions/16981921/relative-imports-in-python-3
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from typing import (
    List, Tuple, Set, Dict, Callable, Any, Optional, Union, Type,
)
from urllib.parse import urljoin

from sparql_ply.components import (
    NodeTerm, PropertyPath, CollectionPath, BlankNodePath, TriplesPath,
    Expression, GraphPattern, Query, ComponentWrapper, QueryComponent,
)
from sparql_ply.nested_replacer import (
    ReplacerNode, ReplacerFactory,
)
from sparql_ply.sparql_yacc import parse as parse_sparql


#########################################################
#
#  Traverse
#
#########################################################

def traverse(
    component: QueryComponent,
    before: Callable[[QueryComponent], None] = lambda x: None,
    after: Callable[[QueryComponent], None] = lambda x: None,
    skip: Callable[[QueryComponent], bool] = lambda x: False,
    prune: Callable[[QueryComponent], bool] = lambda x: False,
):
    '''
    Traverse the component tree.
    '''
    if skip(component):
        return
    before(component)
    if not prune(component):
        if isinstance(component, NodeTerm):
            pass
        elif isinstance(component, (
            PropertyPath, CollectionPath, Expression
        )):
            for child in component.children:
                traverse(child, before, after)
        elif isinstance(component, GraphPattern):
            if component.type & GraphPattern.INLINE_DATA:
                for row in component.children:
                    for child in row:
                        traverse(child, before, after)
            else:
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
                            traverse(t, before, after)
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
    component: QueryComponent, typ: int,
    skip: Callable[[QueryComponent], bool] = lambda x: False,
    prune: Callable[[QueryComponent], bool] = lambda x: False,
) -> List[QueryComponent]:
    '''
    Collect components of specific types.
    '''
    def func(x: QueryComponent):
        nonlocal res
        if x.type & typ:
            res.append(x)

    res = []
    traverse(component, func, skip=skip, prune=prune)
    return res


#########################################################
#
#  Expand Syntax Form
#
#########################################################

class LabelGenerator:
    '''
    Generator for unique labels.
    '''
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
    Class for expanding abbreviations in SPARQL syntax form.
    '''
    def __init__(
        self, sparql: str, 
        default_prefix_dict: Optional[Dict[str, str]] = None,
    ):
        self.sparql = sparql
        self.base_url: Optional[str] = None
        if default_prefix_dict is not None:
            self.prefix_dict = default_prefix_dict
        else:
            self.prefix_dict = dict()
        self.query = parse_sparql(sparql)
        self.collect_prologue(self.query)

        component_list = collect_component(self.query, (
            NodeTerm.TYPE | PropertyPath.TYPE | CollectionPath.TYPE
            | BlankNodePath.TYPE | TriplesPath.TYPE
        ))
        self.blank_gen = LabelGenerator(
            lambda x: f'_:b{x}',
            {
                str(x) for x in component_list
                if isinstance(x, NodeTerm)
                and x.type & NodeTerm.BLANK_NODE_LABEL
            },
        )

        self.key2span: Dict[str, Tuple[int, int]] = dict()
        self.key2component: Dict[str, QueryComponent] = dict()
        self.key2handler: Dict[
            str, Callable[[ReplacerNode], ReplacerNode]
        ] = dict()
        for component in component_list:
            span = (component.lexstart, component.lexstop)
            type_name = get_name_from_type(component.type)
            key = f'{type_name}{span}'
            self.key2span[key] = span
            self.key2component[key] = component
            handler = self.get_handler(component)
            if handler is not None:
                self.key2handler[key] = handler
        
        self.store: Dict[str, List[Tuple[ReplacerNode, ...]]] = dict()
        self.expand_keyword_a      = True
        self.expand_iri            = True
        self.expand_rdf_collection = True
        self.expand_blank_node     = True
        self.expand_pred_obj_list  = True

    # https://www.w3.org/TR/2013/REC-sparql11-query-20130321/#relIRIs
    # https://www.ietf.org/rfc/rfc3986.txt
    # https://github.com/apache/jena/blob/b611d75a1f6f675be182dd9e7aead4e771c231b2/jena-iri/src/main/java/org/apache/jena/iri/impl/ResolvedRelativeIRI.java#L60

    def collect_prologue(self, query: Query):
        if query.prologue is None:
            return
        self.base_url = next((
            str(row[1])[1:-1] for row in query.prologue if len(row) == 2
        ), None)
        for row in query.prologue:
            if len(row) != 3:
                continue
            label = str(row[1])[:-1]
            mapped_url = str(row[2])[1:-1]
            if self.base_url is not None:
                mapped_url = urljoin(self.base_url, mapped_url)
            self.prefix_dict[label] = mapped_url

    def pop_store(self) -> List[Tuple[ReplacerNode, ...]]:
        def key_func(k):
            '''
            Sort the replacer nodes in pre-order.
            '''
            start, end = self.key2span[k]
            return (start, -end, self.rank_key(k))

        sorted_store = sorted(
            [(k, v) for k, v in self.store.items()],
            key=lambda x: key_func(x[0])
        )
        res = list()
        for _, rnodes_list in sorted_store:
            res += rnodes_list
        self.store = dict()
        return res

    def get_builtin_iri(self, abbr: str) -> str:
        abbr = abbr.lower()
        if abbr == 'a':
            local = 'type'
        elif abbr == 'first':
            local = 'first'
        elif abbr == 'rest':
            local = 'rest'
        elif abbr == 'nil':
            local = 'nil'
        else:
            raise ValueError(f'Invalid abbreviation: {abbr}.')

        rdf_label = 'rdf'
        rdf_url = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
        if not self.expand_iri and (
            self.prefix_dict.get(rdf_label, '') == rdf_url
        ):
            return f'{rdf_label}:{local}'
        return f'<{rdf_url}{local}>'
        

    def get_handler(
        self, component: QueryComponent
    ) -> Optional[Callable[[ReplacerNode], ReplacerNode]]:
        if isinstance(component, NodeTerm):
            if component.type & NodeTerm.PREFIXED_NAME:
                return self.prefixed_name_handler
            elif component.type & NodeTerm.IRIREF:
                return self.iriref_handler
            elif component.type & NodeTerm.ANON:
                return self.anon_handler
            elif component.type & NodeTerm.NIL:
                return self.nil_handler
            elif component.type & NodeTerm.SPECIAL and component.value == 'a':
                return self.keyword_a_handler
        elif isinstance(component, CollectionPath):
            return self.rdf_collection_handler
        elif isinstance(component, BlankNodePath):
            return self.blank_node_handler
        elif isinstance(component, TriplesPath):
            return self.triples_path_handler
        return None

    def prefixed_name_handler(self, node: ReplacerNode) -> ReplacerNode:
        if not self.expand_iri:
            return node
        prefixed_name = node.remainder[0]
        label, local = prefixed_name.split(':', 1)
        if label not in self.prefix_dict:
            raise ValueError(f'Prefix {label} not found.')

        res = f'<{self.prefix_dict[label]}{local}>'
        return ReplacerFactory.create_leaf(res, None)

    def iriref_handler(self, node: ReplacerNode) -> ReplacerNode:
        if not self.expand_iri:
            return node
        iriref = node.remainder[0]
        if self.base_url is not None:
            res = '<' + urljoin(self.base_url, iriref[1:-1]) + '>'
        else:
            res = iriref
        return ReplacerFactory.create_leaf(res, None)

    def keyword_a_handler(self, node: ReplacerNode) -> ReplacerNode:
        if not self.expand_keyword_a:
            return node
        kw_a = self.get_builtin_iri('a')
        return ReplacerFactory.create_leaf(kw_a, None)

    def anon_handler(self, node: ReplacerNode) -> ReplacerNode:
        if not self.expand_blank_node:
            return node
        return ReplacerFactory.create_leaf(self.blank_gen(), None)

    def nil_handler(self, node: ReplacerNode) -> ReplacerNode:
        if not self.expand_rdf_collection:
            return node
        rdf_nil = self.get_builtin_iri('nil')
        return ReplacerFactory.create_leaf(rdf_nil, None)

    def rdf_collection_handler(self, node: ReplacerNode) -> ReplacerNode:
        if not self.expand_rdf_collection:
            return node
        rdf_first = self.get_builtin_iri('first')
        rdf_rest = self.get_builtin_iri('rest')
        rdf_nil = self.get_builtin_iri('nil')
        rnode_first = ReplacerFactory.create_leaf(rdf_first, None)
        rnode_rest = ReplacerFactory.create_leaf(rdf_rest, None)
        rnode_nil = ReplacerFactory.create_leaf(rdf_nil, None)

        res = ReplacerFactory.create_leaf(self.blank_gen(), None)
        triples = list()
        curr = res
        for i, x in enumerate(node.children):
            triples.append([curr, rnode_first, x])

            if i == len(node.children) - 1:
                triples.append([curr, rnode_rest, rnode_nil])
            else:
                next = ReplacerFactory.create_leaf(self.blank_gen(), None)
                triples.append([curr, rnode_rest, next])
                curr = next
        self.store[node.key] = [
            [deepcopy(x) for x in triple] for triple in triples
        ]
        return res

    def process_pred_obj_list(
        self, rnode_subj: ReplacerNode, rnode_pol: ReplacerNode,
        pred_obj_list: List[Tuple[
            Union[NodeTerm, PropertyPath],
            List[Union[NodeTerm, CollectionPath, BlankNodePath]],
        ]],
    ) -> List[Tuple[ReplacerNode, ...]]:
        if not self.expand_pred_obj_list:
            return [[rnode_subj, rnode_pol]]

        triples = list()
        i = 0
        for _, obj_list in pred_obj_list:
            rnode_pred = rnode_pol.children[i]
            i += 1
            for _ in obj_list:
                triples.append([
                    deepcopy(rnode_subj), deepcopy(rnode_pred),
                    deepcopy(rnode_pol.children[i]),
                ])
                i += 1
        assert i == len(rnode_pol.children)
        return triples

    def blank_node_handler(self, node: ReplacerNode) -> ReplacerNode:
        if not self.expand_blank_node:
            return node

        res = ReplacerFactory.create_leaf(self.blank_gen(), None)
        rnode_pol = ReplacerNode(
            None, [''] + node.remainder[1:-1] + [''], node.children
        )
        pred_obj_list = self.key2component[node.key].pred_obj_list
        self.store[node.key] = self.process_pred_obj_list(
            deepcopy(res), rnode_pol, pred_obj_list
        )
        return res

    def triples_path_handler(self, node: ReplacerNode) -> ReplacerNode:
        rnode_subj = node.children[0]
        triples_path = self.key2component[node.key]
        assert isinstance(triples_path, TriplesPath)
        subj_unexpanded = (
            isinstance(triples_path.subj, CollectionPath)
            and not self.expand_rdf_collection
        ) or (
            isinstance(triples_path.subj, BlankNodePath)
            and not self.expand_blank_node
        )

        if triples_path.pred_obj_list is None:
            if subj_unexpanded:
                self.store[node.key] = [[rnode_subj]]
        else:
            rnode_pol = ReplacerNode(
                None, [''] + node.remainder[2:], node.children[1:]
            )
            if subj_unexpanded:
                self.store[node.key] = [[rnode_subj, rnode_pol]]
            else:
                self.store[node.key] = self.process_pred_obj_list(
                    rnode_subj, rnode_pol, triples_path.pred_obj_list
                )
        
        remainder, children = list(), list()
        for rnodes in self.pop_store():
            remainder.append('.\n')
            children.append(rnodes[0])
            for rnode in rnodes[1:]:
                remainder.append(' ')
                children.append(rnode)
        remainder.append('')
        remainder[0] = ''
        return ReplacerNode(None, remainder, children)

    def check_replacer_tree(self, rtree: ReplacerNode) -> None:
        def check(node: ReplacerNode):
            component = self.key2component[node.key]
            if isinstance(component, NodeTerm):
                return
            assert cls2count[NodeTerm] == 0, (
                'NodeTerm cannot be a parent of other components.'
            )

            if isinstance(component, (
                PropertyPath, CollectionPath, BlankNodePath
            )):
                assert cls2count[TriplesPath] > 0, (
                    'PropertyPath, CollectionPath and BlankNodePath must be '
                    'descendants of TriplesPath.'
                )

        def dfs(node: ReplacerNode):
            check(node)
            cls = self.key2component[node.key].__class__
            cls2count[cls] += 1
            for child in node.children:
                dfs(child)
            cls2count[cls] -= 1

        cls2count = defaultdict(int)
        (dfs(node) for node in rtree.children)

    def rank_key(self, key: Optional[str]) -> int:
        if key is None:
            return 0
        CLS2HIERARCHY = {
            TriplesPath   : 1,
            BlankNodePath : 2,
            CollectionPath: 3,
            PropertyPath  : 4,
            NodeTerm      : 5,
        }   
        cls = self.key2component[key].__class__
        return CLS2HIERARCHY[cls]

    def run(
        self, expand_keyword_a: bool = True,
        expand_iri: bool = True,
        expand_rdf_collection: bool = True,
        expand_blank_node: bool = True,
        expand_pred_obj_list: Union[bool, str] = ...,
    ) -> str:
        if expand_pred_obj_list is ...:
            if expand_blank_node and expand_rdf_collection:
                expand_pred_obj_list = True
            else:
                expand_pred_obj_list = 'eager'

        if not (
            expand_pred_obj_list is True
            or expand_pred_obj_list is False
        ):
            if (
                isinstance(expand_pred_obj_list, str)
                and expand_pred_obj_list.lower() == 'eager'
            ):
                expand_pred_obj_list = expand_pred_obj_list.lower()
            else:
                raise ValueError(
                    'Invalid value for `expand_pred_obj_list`. '
                    'It must be True, False or "eager".'
                )

        if (
            (not expand_blank_node or not expand_rdf_collection)
            and expand_pred_obj_list is True
        ):
            '''
            For example, `[] ?p ?q1, ?q2` is equivalent to the two triples:
            ```
            _:b57 ?p ?q1 .
            _:b57 ?p ?q2`.
            ```
            But it is not equivalent to the triples:
            ```
            [] ?p ?q1 .
            [] ?p ?q2 .
            ```
            '''
            raise ValueError(
                'Cannot expand predicate-object lists without expanding '
                'blank nodes and RDF collections.'
            )

        self.expand_keyword_a      = bool(expand_keyword_a)
        self.expand_iri            = bool(expand_iri)
        self.expand_rdf_collection = bool(expand_rdf_collection)
        self.expand_blank_node     = bool(expand_blank_node)
        self.expand_pred_obj_list  = expand_pred_obj_list
        self.store = dict()
        
        key2spans = {key: [span] for key, span in self.key2span.items()}
        rt = ReplacerFactory.create_tree(
            self.sparql, key2spans, self.rank_key
        )
        self.check_replacer_tree(rt)
        return rt.replace(self.key2handler).to_str()


def expand_syntax_form(
    sparql: str,
    default_prefix_dict: Optional[Dict[str, str]] = None,
    expand_keyword_a: bool = True,
    expand_iri: bool = True,
    expand_rdf_collection: bool = True,
    expand_blank_node: bool = True,
    expand_pred_obj_list: Union[bool, str] = ...,
) -> str:
    '''
    Expand abbreviations for IRIs and triple patterns.
    https://www.w3.org/TR/2013/REC-sparql11-query-20130321/#sparqlExpandForms
    
    Parameters
    ----------
    sparql: the input SPARQL query string

    default_prefix_dict: the default prefix dictionary

    expand_keyword_a: whether to expand the keyword `a`
    
    expand_iri: whether to expand prefixed names and relative IRIs
    
    expand_rdf_collection: whether to expand RDF collections
    
    expand_blank_node: whether to expand blank nodes
    
    expand_pred_obj_list: whether to expand predicate-object lists and object lists
    
    If `expand_pred_obj_list` is set to "eager", predicate-object lists are
    expanded by default, except when their subjects are blank nodes or RDF
    collections that remain unexpanded.
    
    The default value of `expand_pred_obj_list` is True if both values of
    `expand_blank_node` and `expand_rdf_collection` are True, and "eager"
    otherwise.
    
    Returns
    -------
    The expanded SPARQL query string.
    '''
    expander = SyntaxFormExpander(sparql, default_prefix_dict)
    return expander.run(
        expand_keyword_a, expand_iri, expand_rdf_collection,
        expand_blank_node, expand_pred_obj_list
    )


#########################################################
#
#  Serialize
#
#########################################################

def serialize(
    component: QueryComponent, include_raw_info: bool = False,
) -> Dict[str, Any]:
    '''
    Convert a query component to a JSON-serializable dictionary.
    
    Fields related to raw SPARQL text (e.g. `lexstart`, `lexstop`) are
    included only if `include_raw_info` is True.
    '''

    def post_process(
        component: QueryComponent, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        if include_raw_info:
            return {
                'lexstart': component.lexstart,
                'lexstop': component.lexstop,
                **data,
            }
        return data

    def handle_node_term(component: NodeTerm) -> Dict[str, Any]:
        data = {
            'type': get_name_from_type(component.type),
            'value': component.value,
        }
        if component.language is not None:
            data['language'] = component.language
        if component.datatype is not None:
            data['datatype'] = dfs(component.datatype)
        return post_process(component, data)
    
    def handle_property_path(component: PropertyPath) -> Dict[str, Any]:
        data = {
            'type': get_name_from_type(component.type),
            'operator': component.operator,
            'children': [dfs(x) for x in component.children],
        }
        return post_process(component, data)
    
    def handle_collection_path(component: CollectionPath) -> Dict[str, Any]:
        data = {
            'type': get_name_from_type(component.type),
            'children': [dfs(x) for x in component.children],
        }
        return post_process(component, data)

    def handle_blank_node_path(component: BlankNodePath) -> Dict[str, Any]:
        pred_obj_list = [
            [dfs(p), [dfs(o) for o in ol]] 
            for p, ol in component.pred_obj_list
        ]
        data = {
            'type': get_name_from_type(component.type),
            'pred_obj_list': pred_obj_list,
        }
        return post_process(component, data)

    def handle_triples_path(component: TriplesPath) -> Dict[str, Any]:
        if component.pred_obj_list is not None:
            pred_obj_list = [
                [dfs(p), [dfs(o) for o in ol]]
                for p, ol in component.pred_obj_list
            ]
        else:
            pred_obj_list = None

        data = {
            'type': get_name_from_type(component.type),
            'subj': dfs(component.subj),
            'pred_obj_list': pred_obj_list,
        }
        return post_process(component, data)

    def handle_expression(component: Expression) -> Dict[str, Any]:
        data = {
            'type': get_name_from_type(component.type),
            'operator': component.operator,
            'children': [dfs(x) for x in component.children],
        }
        if component.is_distinct:
            data['is_distinct'] = True
        return post_process(component, data)

    def handle_graph_pattern(component: GraphPattern) -> Dict[str, Any]:
        if component.type & GraphPattern.INLINE_DATA:
            children = [[dfs(x) for x in row] for row in component.children]
        else:
            children = [dfs(x) for x in component.children]

        data = {
            'type': get_name_from_type(component.type),
            'children': children,
        }
        if component.is_silent:
            data['is_silent'] = True
        return post_process(component, data)

    def handle_query(component: Query) -> Dict[str, Any]:       
        data = dict()
        data['type'] = get_name_from_type(component.type)

        if component.prologue is not None:
            data['prologue'] = [
                [str(row[0])] + [dfs(x) for x in row[1:]]
                for row in component.prologue
            ]
        if component.select_modifier is not None:
            data['select_modifier'] = str(component.select_modifier)
        if component.target is not None:
            if isinstance(component.target, (NodeTerm, GraphPattern)):
                target = dfs(component.target)
            elif isinstance(component.target, Sequence):
                target = list()
                for t in component.target:
                    if isinstance(t, NodeTerm):
                        target.append(dfs(t))
                    elif isinstance(t, ComponentWrapper):
                        target.append([dfs(t[0]), dfs(t[1])])
                    else:
                        raise NotImplementedError
            else:
                raise NotImplementedError
            data['target'] = target
        if component.dataset is not None:
            data['dataset'] = [
                [str(x) for x in d[:-1]] + [dfs(d[-1])]
                for d in component.dataset
            ]
        if component.pattern is not None:
            data['pattern'] = dfs(component.pattern)

        if component.group_by is not None:
            group_by = list()
            for x in component.group_by:
                if isinstance(x, (NodeTerm, Expression)):
                    group_by.append(dfs(x))
                elif isinstance(x, ComponentWrapper):
                    group_by.append([dfs(x[0]), dfs(x[1])])
                else:
                    raise NotImplementedError
            data['group_by'] = group_by
        if component.having is not None:
            data['having'] = [dfs(x) for x in component.having]
        if component.order_by is not None:
            order_by = list()
            for x in component.order_by:
                if isinstance(x, (NodeTerm, Expression)):
                    order_by.append(dfs(x))
                elif isinstance(x, ComponentWrapper):
                    order_by.append([str(x[0]), dfs(x[1])])
                else:
                    raise NotImplementedError
            data['order_by'] = order_by
        if component.limit is not None:
            data['limit'] = dfs(component.limit.value)
        if component.offset is not None:
            data['offset'] = dfs(component.offset.value)
        if component.values is not None:
            data['values'] = [
                [dfs(x) for x in row] for row in component.values
            ]

        if include_raw_info:
            span_dict = dict()
            if component.prologue is not None:
                span_dict['prologue'] = [
                    [x.lexstart, x.lexstop] for x in component.prologue
                ]
            if component.select_modifier is not None:
                span_dict['select_modifier'] = [
                    component.select_modifier.lexstart,
                    component.select_modifier.lexstop,
                ]
            if component.target is not None:
                if isinstance(component.target, Sequence):
                    span_dict['target'] = [
                        [t.lexstart, t.lexstop] for t in component.target
                        if isinstance(t, ComponentWrapper)
                    ]
            if component.dataset is not None:
                span_dict['dataset'] = [
                    [x.lexstart, x.lexstop] for x in component.dataset
                ]
            if component.group_by is not None:
                span_dict['group_by'] = [
                    [component.group_by.lexstart, component.group_by.lexstop]
                ] + [
                    [x.lexstart, x.lexstop] for x in component.group_by
                    if isinstance(x, ComponentWrapper)
                ]
            if component.having is not None:
                span_dict['having'] = [
                    component.having.lexstart, component.having.lexstop
                ]
            if component.order_by is not None:
                span_dict['order_by'] = [
                    [component.order_by.lexstart, component.order_by.lexstop]
                ] + [
                    [x.lexstart, x.lexstop] for x in component.order_by
                    if isinstance(x, ComponentWrapper)
                ]
            if component.limit is not None:
                span_dict['limit'] = [
                    component.limit.lexstart, component.limit.lexstop
                ]
            if component.offset is not None:
                span_dict['offset'] = [
                    component.offset.lexstart, component.offset.lexstop
                ]
            if component.values is not None:
                span_dict['values'] = [
                    component.values.lexstart, component.values.lexstop
                ]
            data['wrapper_span'] = span_dict

        return post_process(component, data)

    def dfs(component: QueryComponent) -> Dict[str, Any]:
        typ = component.type
        if typ & NodeTerm.TYPE:
            return handle_node_term(component)
        elif typ & PropertyPath.TYPE:
            return handle_property_path(component)
        elif typ & CollectionPath.TYPE:
            return handle_collection_path(component)
        elif typ & BlankNodePath.TYPE:
            return handle_blank_node_path(component)
        elif typ & TriplesPath.TYPE:
            return handle_triples_path(component)
        elif typ & Expression.TYPE:
            return handle_expression(component)
        elif typ & GraphPattern.TYPE:
            return handle_graph_pattern(component)
        elif typ & Query.TYPE:
            return handle_query(component)
        else:
            raise NotImplementedError(
                f'[serialize] Unsupported type: {typ}.'
            )
    
    if include_raw_info:
        if (
            (not isinstance(component, Query))
            or component.type & Query.SUB_SELECT
        ):
            raise ValueError(
                'Raw information is not available for '
                f'{get_name_from_type(component.type)}.'
            )
    
    data = dfs(component)
    if include_raw_info:
        data['raw_sparql'] = component.raw_sparql
    return data


def deserialize(
    data: Dict[str, Any],
    include_raw_info: bool = False,
) -> QueryComponent:
    '''
    Convert a JSON-serializable dictionary to a query component.
    
    Fields related to raw SPARQL text (e.g. `lexstart`, `lexstop`) are
    available in the output component only if `include_raw_info` is True.
    '''
    
    def get_lex_span(data: Dict[str, Any]) -> Tuple[int, int]:
        if include_raw_info:
            return data['lexstart'], data['lexstop']
        return -1, -1
    
    def to_node_term(data: Dict[str, Any]) -> NodeTerm:
        lexstart, lexstop = get_lex_span(data)
        typ = get_type_from_name(data['type'])
        language = data.get('language', None)
        if 'datatype' in data:
            datatype = dfs(data['datatype'])
        else:
            datatype = None

        return NodeTerm(
            lexstart, lexstop, data['value'],
            typ, language, datatype,
        )
    
    def to_property_path(data: Dict[str, Any]) -> PropertyPath:
        lexstart, lexstop = get_lex_span(data)
        return PropertyPath(
            lexstart, lexstop, [dfs(x) for x in data['children']],
            data['operator'],
        )
    
    def to_collection_path(data: Dict[str, Any]) -> CollectionPath:
        lexstart, lexstop = get_lex_span(data)
        return CollectionPath(
            lexstart, lexstop, [dfs(x) for x in data['children']],
        )

    def to_blank_node_path(data: Dict[str, Any]) -> BlankNodePath:
        lexstart, lexstop = get_lex_span(data)
        pred_obj_list = [
            [dfs(p), [dfs(o) for o in ol]]
            for p, ol in data['pred_obj_list']
        ]
        return BlankNodePath(lexstart, lexstop, pred_obj_list)

    def to_triples_path(data: Dict[str, Any]) -> TriplesPath:
        lexstart, lexstop = get_lex_span(data)
        subj = dfs(data['subj'])
        if data['pred_obj_list'] is not None:
            pred_obj_list = [
                [dfs(p), [dfs(o) for o in ol]]
                for p, ol in data['pred_obj_list']
            ]
        else:
            pred_obj_list = None

        return TriplesPath(lexstart, lexstop, subj, pred_obj_list)

    def to_expression(data: Dict[str, Any]) -> Expression:
        lexstart, lexstop = get_lex_span(data)
        return Expression(
            lexstart, lexstop, [dfs(x) for x in data['children']],
            data['operator'], data.get('is_distinct', False),
        )

    def to_graph_pattern(data: Dict[str, Any]) -> GraphPattern:
        lexstart, lexstop = get_lex_span(data)
        typ = get_type_from_name(data['type'])
        if typ & GraphPattern.INLINE_DATA:
            children = [[dfs(x) for x in row] for row in data['children']]
        else:
            children = [dfs(x) for x in data['children']]

        is_silent = data.get('is_silent', False)
        return GraphPattern(lexstart, lexstop, children, typ, is_silent)

    def to_query(data: Dict[str, Any]) -> Query:
        prologue        = None
        select_modifier = None
        target          = None
        dataset         = None
        pattern         = None
        group_by        = None
        having          = None
        order_by        = None
        limit           = None
        offset          = None
        values          = None

        if 'prologue' in data:
            prologue = list()
            for row in data['prologue']:
                decl = [str(row[0])] + [dfs(x) for x in row[1:]]
                prologue.append(ComponentWrapper(-1, -1, decl))
        if 'select_modifier' in data:
            select_modifier = ComponentWrapper(
                -1, -1, data['select_modifier'],
            )
        if 'target' in data:
            if isinstance(data['target'], dict):
                target = dfs(data['target'])
            elif isinstance(data['target'], list):
                target = list()
                for t in data['target']:
                    if isinstance(t, dict):
                        target.append(dfs(t))
                    elif isinstance(t, list):
                        target.append(ComponentWrapper(
                            -1, -1, [dfs(t[0]), dfs(t[1])]
                        ))
                    else:
                        raise NotImplementedError
            else:
                raise NotImplementedError
        if 'dataset' in data:
            dataset = list()
            for row in data['dataset']:
                decl = [str(x) for x in row[:-1]] + [dfs(row[-1])]
                dataset.append(ComponentWrapper(-1, -1, decl))
        if 'pattern' in data:
            pattern = dfs(data['pattern'])
        if 'group_by' in data:
            conditions = list()
            for x in data['group_by']:
                if isinstance(x, dict):
                    conditions.append(dfs(x))
                elif isinstance(x, list):
                    conditions.append(ComponentWrapper(
                        -1, -1, [dfs(x[0]), dfs(x[1])]
                    ))
                else:
                    raise NotImplementedError
            group_by = ComponentWrapper(-1, -1, conditions)
        if 'having' in data:
            conditions = [dfs(x) for x in data['having']]
            having = ComponentWrapper(-1, -1, conditions)
        if 'order_by' in data:
            conditions = list()
            for x in data['order_by']:
                if isinstance(x, dict):
                    conditions.append(dfs(x))
                elif isinstance(x, list):
                    conditions.append(ComponentWrapper(
                        -1, -1, [str(x[0]), dfs(x[1])]
                    ))
                else:
                    raise NotImplementedError
            order_by = ComponentWrapper(-1, -1, conditions)
        if 'limit' in data:
            limit = ComponentWrapper(-1, -1, dfs(data['limit']))
        if 'offset' in data:
            offset = ComponentWrapper(-1, -1, dfs(data['offset']))
        if 'values' in data:
            data_block = [
                [dfs(x) for x in row] for row in data['values']
            ]
            values = ComponentWrapper(-1, -1, data_block)

        if include_raw_info:
            span_dict = data['wrapper_span']
            if prologue is not None:
                for i, row in enumerate(prologue):
                    row.lexstart, row.lexstop = span_dict['prologue'][i]
            if select_modifier is not None:
                select_modifier.lexstart, select_modifier.lexstop = (
                    span_dict['select_modifier']
                )
            if target is not None:
                if isinstance(target, Sequence):
                    spans = span_dict['target'].reverse()
                    for t in target:
                        if isinstance(t, ComponentWrapper):
                            t.lexstart, t.lexstop = spans.pop()
            if dataset is not None:
                for i, d in enumerate(dataset):
                    d.lexstart, d.lexstop = span_dict['dataset'][i]
            if group_by is not None:
                group_by.lexstart, group_by.lexstop = span_dict['group_by'][0]
                spans = span_dict['group_by'][1:].reverse()
                for i, x in enumerate(group_by):
                    if isinstance(x, ComponentWrapper):
                        x.lexstart, x.lexstop = spans.pop()
            if having is not None:
                having.lexstart, having.lexstop = span_dict['having']
            if order_by is not None:
                order_by.lexstart, order_by.lexstop = span_dict['order_by'][0]
                spans = span_dict['order_by'][1:].reverse()
                for i, x in enumerate(order_by):
                    if isinstance(x, ComponentWrapper):
                        x.lexstart, x.lexstop = spans.pop()
            if limit is not None:
                limit.lexstart, limit.lexstop = span_dict['limit']
            if offset is not None:
                offset.lexstart, offset.lexstop = span_dict['offset']
            if values is not None:
                values.lexstart, values.lexstop = span_dict['values']

        lexstart, lexstop = get_lex_span(data)
        return Query(
            lexstart, lexstop, get_type_from_name(data['type']),
            prologue=prologue,
            select_modifier=select_modifier,
            target=target,
            dataset=dataset,
            pattern=pattern,
            group_by=group_by,
            having=having,
            order_by=order_by,
            limit=limit,
            offset=offset,
            values=values,
        )

    def dfs(data: Dict[str, Any]) -> QueryComponent:
        typ = get_type_from_name(data['type'])
        if typ & NodeTerm.TYPE:
            return to_node_term(data)
        elif typ & PropertyPath.TYPE:
            return to_property_path(data)
        elif typ & CollectionPath.TYPE:
            return to_collection_path(data)
        elif typ & BlankNodePath.TYPE:
            return to_blank_node_path(data)
        elif typ & TriplesPath.TYPE:
            return to_triples_path(data)
        elif typ & Expression.TYPE:
            return to_expression(data)
        elif typ & GraphPattern.TYPE:
            return to_graph_pattern(data)
        elif typ & Query.TYPE:
            return to_query(data)
        else:
            raise NotImplementedError(
                f'[deserialize] Unsupported type: {typ}.'
            )

    typ = get_type_from_name(data['type'])
    if include_raw_info:
        if (not typ & Query.TYPE) or typ & Query.SUB_SELECT:
            raise ValueError(
                'Raw information is not available for '
                f'{get_name_from_type(typ)}.'
            )
        if 'raw_sparql' not in data:
            raise ValueError('Raw information is not found.')

    component = dfs(data)
    if include_raw_info:
        component.raw_sparql = data['raw_sparql']
    return component


#########################################################
#
#  Component Type Information
#
#########################################################

COMPONENT_TYPE_INFO: List[Tuple[
    Type[QueryComponent], int, str, str
]]= [
    (NodeTerm, NodeTerm.RDF_LITERAL, 'NodeTerm', 'RDF_LITERAL'),
    (NodeTerm, NodeTerm.BOOLEAN, 'NodeTerm', 'BOOLEAN'),
    (NodeTerm, NodeTerm.INTEGER, 'NodeTerm', 'INTEGER'),
    (NodeTerm, NodeTerm.DECIMAL, 'NodeTerm', 'DECIMAL'),
    (NodeTerm, NodeTerm.DOUBLE, 'NodeTerm', 'DOUBLE'),
    (NodeTerm, NodeTerm.IRIREF, 'NodeTerm', 'IRIREF'),
    (NodeTerm, NodeTerm.PREFIXED_NAME, 'NodeTerm', 'PREFIXED_NAME'),
    (NodeTerm, NodeTerm.BLANK_NODE_LABEL, 'NodeTerm', 'BLANK_NODE_LABEL'),
    (NodeTerm, NodeTerm.ANON, 'NodeTerm', 'ANON'),
    (NodeTerm, NodeTerm.NIL, 'NodeTerm', 'NIL'),
    (NodeTerm, NodeTerm.VAR, 'NodeTerm', 'VAR'),
    (NodeTerm, NodeTerm.SPECIAL, 'NodeTerm', 'SPECIAL'),
    (PropertyPath, PropertyPath.NOP, 'PropertyPath', 'NOP'),
    (PropertyPath, PropertyPath.UNARY_PREFIX, 'PropertyPath', 'UNARY_PREFIX'),
    (PropertyPath, PropertyPath.UNARY_POSTFIX, 'PropertyPath', 'UNARY_POSTFIX'),
    (PropertyPath, PropertyPath.BINARY_OP, 'PropertyPath', 'BINARY_OP'),
    (CollectionPath, CollectionPath.TYPE, 'CollectionPath', 'COLLECTION_PATH'),
    (BlankNodePath, BlankNodePath.TYPE, 'BlankNodePath', 'BLANK_NODE_PATH'),
    (TriplesPath, TriplesPath.TYPE, 'TriplesPath', 'TRIPLES_PATH'),
    (Expression, Expression.NOP, 'Expression', 'NOP'),
    (Expression, Expression.UNARY_OP, 'Expression', 'UNARY_OP'),
    (Expression, Expression.BINARY_LOGICAL, 'Expression', 'BINARY_LOGICAL'),
    (Expression, Expression.BINARY_COMPARISON, 'Expression', 'BINARY_COMPARISON'),
    (Expression, Expression.BINARY_ARITHMETIC, 'Expression', 'BINARY_ARITHMETIC'),
    (Expression, Expression.IN_OP, 'Expression', 'IN_OP'),
    (Expression, Expression.FUNC_FORM, 'Expression', 'FUNC_FORM'),
    (Expression, Expression.FUNC_TERM, 'Expression', 'FUNC_TERM'),
    (Expression, Expression.FUNC_STR, 'Expression', 'FUNC_STR'),
    (Expression, Expression.FUNC_NUMERIC, 'Expression', 'FUNC_NUMERIC'),
    (Expression, Expression.FUNC_TIME, 'Expression', 'FUNC_TIME'),
    (Expression, Expression.FUNC_HASH, 'Expression', 'FUNC_HASH'),
    (Expression, Expression.AGGREGATE, 'Expression', 'AGGREGATE'),
    (Expression, Expression.IRI_FUNC, 'Expression', 'IRI_FUNC'),
    (GraphPattern, GraphPattern.TRIPLES_BLOCK, 'GraphPattern', 'TRIPLES_BLOCK'),
    (GraphPattern, GraphPattern.GROUP, 'GraphPattern', 'GROUP'),
    (GraphPattern, GraphPattern.UNION, 'GraphPattern', 'UNION'),
    (GraphPattern, GraphPattern.OPTIONAL, 'GraphPattern', 'OPTIONAL'),
    (GraphPattern, GraphPattern.MINUS, 'GraphPattern', 'MINUS'),
    (GraphPattern, GraphPattern.GRAPH, 'GraphPattern', 'GRAPH'),
    (GraphPattern, GraphPattern.SERVICE, 'GraphPattern', 'SERVICE'),
    (GraphPattern, GraphPattern.FILTER, 'GraphPattern', 'FILTER'),
    (GraphPattern, GraphPattern.BIND, 'GraphPattern', 'BIND'),
    (GraphPattern, GraphPattern.INLINE_DATA, 'GraphPattern', 'INLINE_DATA'),
    (GraphPattern, GraphPattern.SUB_SELECT, 'GraphPattern', 'SUB_SELECT'),
    (Query, Query.SELECT, 'Query', 'SELECT'),
    (Query, Query.CONSTRUCT, 'Query', 'CONSTRUCT'),
    (Query, Query.DESCRIBE, 'Query', 'DESCRIBE'),
    (Query, Query.ASK, 'Query', 'ASK'),
    (Query, Query.SUB_SELECT, 'Query', 'SUB_SELECT'),
]


COMPONENT_TYPE_DICT = {
    typ: (cls, typ, cls_str, sub_type_str)
    for cls, typ, cls_str, sub_type_str in COMPONENT_TYPE_INFO
}

COMPONENT_NAME_DICT = {
    f'{cls_str}.{sub_type_str}': (cls, typ, cls_str, sub_type_str)
    for cls, typ, cls_str, sub_type_str in COMPONENT_TYPE_INFO
}


def get_type_from_name(name: str) -> int:
    _, typ, _, _ = COMPONENT_NAME_DICT[name]
    return typ


def get_name_from_type(typ: int) -> str:
    _, _, cls_str, sub_type_str = COMPONENT_TYPE_DICT[typ]
    return f'{cls_str}.{sub_type_str}'


def get_class_from_type(typ: int) -> Type[QueryComponent]:
    cls, _, _, _ = COMPONENT_TYPE_DICT[typ]
    return cls


#########################################################
#
#  Get Variable Scopes
#
#########################################################

def get_variable_scopes(
    component: QueryComponent
) -> List[List[Tuple[int, int]]]:
    '''
    Get the variable scopes of a query component.

    Parameters
    ----------
    component: QueryComponent

    Returns
    -------
    A list of variable scopes. Each scope is a list of integer pairs
    representing the lexical start and end positions of the variable 
    occurrences. The integer pairs within each scope are sorted in
    sequential order.
    '''
    stack_var2scope: List[
        Dict[str, List[Tuple[int, int]]]
    ] = [defaultdict(list)]
    scopes: List[List[Tuple[int, int]]] = list()

    def before(component: QueryComponent) -> None:
        if (
            isinstance(component, NodeTerm)
            and component.type == NodeTerm.VAR
        ):
            stack_var2scope[-1][component.value].append(
                [component.lexstart, component.lexstop]
            )
        elif (
            isinstance(component, Query)
            and component.type & Query.SUB_SELECT
        ):
            stack_var2scope.append(defaultdict(list))

    def after(component: QueryComponent) -> None:
        if not (
            isinstance(component, Query)
            and component.type & Query.SUB_SELECT
        ):
            return
        
        var2scope = stack_var2scope.pop()
        if isinstance(component.target, NodeTerm):
            # SELECT * WHERE { ... }
            stack_var2scope[-1].update(var2scope)
        elif isinstance(component.target, Sequence):
            tar_vars = set()
            for t in component.target:
                if isinstance(t, NodeTerm):
                    tar_vars.add(t.value)
                elif isinstance(t, ComponentWrapper):
                    tar_vars.add(t[1].value)
                else:
                    raise NotImplementedError
            for v, scope in var2scope.items():
                if v in tar_vars:
                    stack_var2scope[-1][v].extend(scope)
                else:
                    scopes.append(scope)
        else:
            raise NotImplementedError

    traverse(component, before, after)
    scopes.extend(stack_var2scope[-1].values())
    return [sorted(scope) for scope in scopes]


if __name__ == '__main__':
    sparql = (
        'PREFIX  rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n'
        'PREFIX  wde:  <http://www.wikidata.org/entity/>\n'
        'SELECT * WHERE  {\n'
        '(?a ?b ?c ?d) .\n'
        '[?p1 ?q1; ?ps ("abc" wde:Q1234)]\n'
        '    rdf:abc _:b0, _:b1 ;\n'
        '    rdf:def [?p2 [?p3 [?p4 _:b4]]] .\n'
        '[?p5 ?q5].\n'
        'wde:Q5678 ?p6 ?q6; a ?q7, [].\n'
        '}'
    )
    print(expand_syntax_form(
        sparql,
        expand_keyword_a=False,
        expand_iri=False,
        # expand_rdf_collection=False,
        # expand_blank_node=False,
        # expand_pred_obj_list=False,
    ))
    
    # sparql = 'SELECT * WHERE { BIND (42 AS ?s) }'
    # q1 = parse_sparql(sparql)
    # d1 = serialize(q1)
    # q2 = deserialize(d1)
    # d2 = serialize(q2)
    # assert q1.to_str() == q2.to_str()
    # assert d1 == d2
    # import json
    # print(json.dumps(d2, indent=2))

    # sparql = (
    #     'SELECT ?a ?b ?c ?d {'
    #     '?a :p ?b. ?a :q ?c.'
    #     '{ SELECT (?c AS ?d) { ?a :p :e. ?a :r ?c.} }'
    #     '{ SELECT ?a { ?a :q ?c. { SELECT ?c { ?c :r :e.} } } }'
    #     '}'
    # )
    # q = parse_sparql(sparql)
    # for i in range(0, len(sparql), 25):
    #     print(f'[{i:0{len(str(len(sparql)))}d}]', sparql[i:i + 25])
    # for scope in get_variable_scopes(q):
    #     print(sparql[scope[0][0]:scope[0][1]], scope)

