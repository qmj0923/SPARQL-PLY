'''
Copyright (c) 2024 qmj0923
https://github.com/qmj0923/SPARQL-PLY
'''

from __future__ import annotations
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from typing import (
    List, Tuple, Optional, Union, TypeVar, Generic,
)


def gen_type() -> int:
    try:
        bit = gen_type.__bit
    except AttributeError:
        gen_type.__bit = itertools.count(0)
        bit = gen_type.__bit
    return 1 << next(bit)


class QueryComponent(ABC):
    '''
    Abstract base class for all components of a SPARQL query.
    '''
    DEBUG = False

    def __init__(self, lexstart: int, lexstop: int, typ: int):
        self.lexstart = lexstart
        self.lexstop = lexstop
        self.type = typ
        self._check()

    @abstractmethod
    def to_str(self) -> str:
        raise NotImplementedError(
            f'to_str is not implemented for {self.__class__}'
        )

    def __repr__(self) -> str:
        return self.to_str()

    def __str__(self) -> str:
        return self.to_str()
    
    def __eq__(self, other) -> bool:
        raise NotImplementedError(
            f'__eq__ is not implemented for {self.__class__}'
        )
    
    def __hash__(self) -> int:
        raise NotImplementedError(
            f'__hash__ is not implemented for {self.__class__}'
        )

    def _check(self):
        if not self.DEBUG:
            return

        print(flush=True)
        print(self.__class__.__name__, self.type)
        print(self.lexstart, self.lexstop)
        print(self.to_str())
        print(flush=True)


#########################################################
#
#  NodeTerm
#
#  (leaf nodes of the parse tree)
#
#########################################################

class NodeTerm(QueryComponent):
    '''
    GraphTerm | Var | 'a' | '*' | 'UNDEF'
    '''
    RDF_LITERAL = gen_type()  # 'RDFLiteral'
    BOOLEAN     = gen_type()  # 'BooleanLiteral'
    INTEGER     = gen_type()  # 'xsd_integer'
    DECIMAL     = gen_type()  # 'xsd_decimal'
    DOUBLE      = gen_type()  # 'xsd_double'
    
    IRIREF           = gen_type()  # 'IRIREF'
    PREFIXED_NAME    = gen_type()  # 'PrefixedName'
    BLANK_NODE_LABEL = gen_type()  # 'BLANK_NODE_LABEL'
    ANON             = gen_type()  # 'ANON'
    NIL              = gen_type()  # 'NIL'

    VAR     = gen_type()  # 'Var'
    SPECIAL = gen_type()  # 'a', '*', 'UNDEF'

    NUMERIC = INTEGER | DECIMAL | DOUBLE
    IRI = IRIREF | PREFIXED_NAME 
    BLANK_NODE = BLANK_NODE_LABEL | ANON
    
    # RDF_TERM = IRI | RDF_LITERAL | BLANK_NODE
    # GRAPH_TERM = IRI | RDF_LITERAL | NUMERIC | BOOLEAN | BLANK_NODE | NIL
    
    TYPE = (
        RDF_LITERAL | BOOLEAN | INTEGER | DECIMAL | DOUBLE | IRIREF
        | PREFIXED_NAME | BLANK_NODE_LABEL | ANON | NIL | VAR | SPECIAL
    )
    
    def __init__(
        self, lexstart: int, lexstop: int,
        content: str, typ: int,
        language: Optional[str] = None,
        datatype: Optional[NodeTerm] = None,
    ):
        if typ & NodeTerm.BOOLEAN:
            self.value = content.lower()
        elif typ & NodeTerm.ANON:
            self.value = '[]'
        elif typ & NodeTerm.NIL:
            self.value = '()'
        elif typ & NodeTerm.VAR:
            self.value = '?' + content[1:]
        elif typ & NodeTerm.SPECIAL and content.upper() == 'UNDEF':
            self.value = 'UNDEF'
        else:
            self.value = content

        if typ & NodeTerm.RDF_LITERAL:
            self.language = language
            self.datatype = datatype
        else:
            self.language = None
            self.datatype = None
        
        super().__init__(lexstart, lexstop, typ)

    def to_str(self):
        res = self.value
        if self.type & NodeTerm.RDF_LITERAL:
            if self.language is not None:
                res += '@' + self.language
            if self.datatype is not None:
                res += '^^' + str(self.datatype)
        return res

    def __eq__(self, other):
        return (
            isinstance(other, NodeTerm)
            and self.value == other.value
            and self.type == other.type
            and self.language == self.language
            and self.datatype == self.datatype
        )

    def __hash__(self):
        return hash(self.value) ^ hash(self.type)


#########################################################
#
#  PropertyPath 
#
#########################################################

class PropertyPath(QueryComponent):
    '''
    The PropertyPath class represents a node of the parse tree that
    corresponds to a property path in a SPARQL query.
    
    Each leaf node of the parse tree is a `NodeTerm` object, and
    it can be either an IRI or the keyword 'a'.
    '''
    NOP           = gen_type()  # No operation (for bracketed path)
    UNARY_PREFIX  = gen_type()  # !, ^
    UNARY_POSTFIX = gen_type()  # ?, *, +
    BINARY_OP     = gen_type()  # |, /

    UNARY_OP = UNARY_PREFIX | UNARY_POSTFIX
    TYPE = NOP | UNARY_PREFIX | UNARY_POSTFIX | BINARY_OP

    def __init__(
        self, lexstart: int, lexstop: int,
        children: List[Union[PropertyPath, NodeTerm]],
        operator: Optional[str] = None,
    ):
        self.children = children
        self.operator = operator
        
        if self.operator is None:
            assert len(self.children) == 1
            typ = PropertyPath.NOP
        elif self.operator in ['^', '!']:
            assert len(self.children) == 1
            typ = PropertyPath.UNARY_PREFIX
        elif self.operator in ['?', '*', '+']:
            assert len(self.children) == 1
            typ = PropertyPath.UNARY_POSTFIX
        elif self.operator in ['|', '/']:
            assert len(self.children) >= 2
            typ = PropertyPath.BINARY_OP
        else:
            raise NotImplementedError

        super().__init__(lexstart, lexstop, typ)

    def to_str(self):
        if self.type & PropertyPath.BINARY_OP:
            return self.operator.join(
                [str(x) for x in self.children]
            )
        val = self.children[0]
        if self.type & PropertyPath.NOP:
            return f'({val})'
        elif self.type & PropertyPath.UNARY_PREFIX:
            return f'{self.operator}{val}'
        elif self.type & PropertyPath.UNARY_POSTFIX:
            return f'{val}{self.operator}'
        else:
            raise NotImplementedError


#########################################################
#
#  CollectionPath
#
#########################################################

class CollectionPath(QueryComponent):
    '''
    [102] Collection & [103] CollectionPath
    '''
    TYPE = gen_type()

    def __init__(
        self, lexstart: int, lexstop: int,
        children: List[Union[NodeTerm, CollectionPath, BlankNodePath]]
    ):
        self.children = children
        super().__init__(lexstart, lexstop, CollectionPath.TYPE)

    def to_str(self):
        return '(' + ' '.join([str(x) for x in self.children]) + ')'


#########################################################
#
#  BlankNodePath
#
#########################################################

class BlankNodePath(QueryComponent):
    '''
    [99]  BlankNodePropertyList & [101] BlankNodePropertyListPath
    '''
    TYPE = gen_type()

    def __init__(
        self, lexstart: int, lexstop: int,
        pred_obj_list: List[Tuple[
            Union[NodeTerm, PropertyPath],
            List[Union[NodeTerm, CollectionPath, BlankNodePath]],
        ]],
    ):
        self.pred_obj_list = pred_obj_list
        super().__init__(lexstart, lexstop, BlankNodePath.TYPE)

    def to_str(self):
        po = '; '.join([
            f'{p} ' + ', '.join([str(o) for o in ol])
            for p, ol in self.pred_obj_list
        ])
        return '[' + po + ']'


#########################################################
#
#  TriplesPath
#
#########################################################

class TriplesPath(QueryComponent):
    '''
    [81] TriplesSameSubjectPath
    '''
    TYPE = gen_type()

    def __init__(
        self, lexstart: int, lexstop: int,
        subj: Union[NodeTerm, CollectionPath, BlankNodePath], 
        pred_obj_list: Optional[List[Tuple[
            Union[NodeTerm, PropertyPath],
            List[Union[NodeTerm, CollectionPath, BlankNodePath]],
        ]]],
    ):
        self.subj = subj
        self.pred_obj_list = pred_obj_list
        if self.pred_obj_list is None:
            assert isinstance(self.subj, (CollectionPath, BlankNodePath))

        super().__init__(lexstart, lexstop, TriplesPath.TYPE)

    def to_str(self):
        if self.pred_obj_list is None:
            return str(self.subj)

        po = '; '.join([
            f'{p} ' + ', '.join([str(o) for o in ol])
            for p, ol in self.pred_obj_list
        ])
        return str(self.subj) + ' ' + po


#########################################################
#
#  Expression
#
#########################################################

class Expression(QueryComponent):
    '''
    [110] Expression
    '''
    NOP               = gen_type()  # No operation (for bracketed expression)
    UNARY_OP          = gen_type()  # +, -, !
    BINARY_LOGICAL    = gen_type()  # ||, &&
    BINARY_COMPARISON = gen_type()  # =, !=, <, >, <=, >=
    BINARY_ARITHMETIC = gen_type()  # +, -, *, /
    IN_OP             = gen_type()  # IN, NOT IN
    
    FUNC_FORM    = gen_type()  # Functional Forms (not operators)
    FUNC_TERM    = gen_type()  # Functions on RDF Terms
    FUNC_STR     = gen_type()  # Functions on Strings
    FUNC_NUMERIC = gen_type()  # Functions on Numerics
    FUNC_TIME    = gen_type()  # Functions on Dates and Times
    FUNC_HASH    = gen_type()  # Hash Functions
    AGGREGATE    = gen_type()  # Aggregate Functions
    IRI_FUNC     = gen_type()  # Functions named by an IRI
    
    BINARY_OP    = BINARY_LOGICAL | BINARY_COMPARISON | BINARY_ARITHMETIC
    BUILTIN_CALL = (
        FUNC_FORM | FUNC_TERM | FUNC_STR | FUNC_NUMERIC | FUNC_TIME
        | FUNC_HASH | AGGREGATE
    )
    CALL         = BUILTIN_CALL | IRI_FUNC

    TYPE = (
        NOP | UNARY_OP | BINARY_LOGICAL | BINARY_COMPARISON
        | BINARY_ARITHMETIC | IN_OP | FUNC_FORM | FUNC_TERM | FUNC_STR
        | FUNC_NUMERIC | FUNC_TIME | FUNC_HASH | AGGREGATE | IRI_FUNC
    )

    TYPE2FUNCTIONS = {
        FUNC_FORM: [
            'BOUND', 'COALESCE', 'IF', 'SAMETERM',
            'EXISTS', 'NOT EXISTS',
        ],
        FUNC_TERM: [
            'STR', 'LANG', 'DATATYPE', 'IRI', 'URI', 'BNODE',
            'UUID', 'STRUUID', 'STRLANG', 'STRDT', 'ISIRI', 'ISURI',
            'ISBLANK', 'ISLITERAL', 'ISNUMERIC',
        ],
        FUNC_STR: [
            'LANGMATCHES', 'CONCAT', 'SUBSTR', 'STRLEN', 'REPLACE',
            'UCASE', 'LCASE', 'ENCODE_FOR_URI', 'CONTAINS', 
            'STRSTARTS', 'STRENDS', 'STRBEFORE', 'STRAFTER',
            'REGEX',     
        ],
        FUNC_NUMERIC:[
            'RAND', 'ABS', 'CEIL', 'FLOOR', 'ROUND',
        ],
        FUNC_TIME:[
            'YEAR', 'MONTH', 'DAY', 'HOURS', 'MINUTES', 'SECONDS',
            'TIMEZONE', 'TZ', 'NOW',
        ],
        FUNC_HASH: [
            'MD5', 'SHA1', 'SHA256', 'SHA384', 'SHA512',
        ],
        AGGREGATE:[
            'COUNT', 'SUM', 'MIN', 'MAX', 'AVG', 'SAMPLE', 'GROUP_CONCAT',
        ],
    }
    FUNC2TYPE = {
        func: typ 
        for typ, funcs in TYPE2FUNCTIONS.items()
        for func in funcs
    }

    def __init__(
        self, lexstart: int, lexstop: int,
        children: List[Union[NodeTerm, Expression, GraphPattern]],
        operator: Optional[str] = None,
        is_distinct: bool = False,
    ):
        self.children = children
        self.operator = operator.upper() if operator else None
        self.is_distinct = is_distinct
        
        if self.operator is None:
            assert len(self.children) == 1
            typ = Expression.NOP
        elif (
            self.operator in ['+', '-', '!']
            and len(self.children) == 1
        ):
            typ = Expression.UNARY_OP
        elif self.operator in ['||', '&&']:
            typ = Expression.BINARY_LOGICAL
        elif self.operator in ['=', '!=', '<', '>', '<=', '>=']:
            assert len(self.children) == 2
            typ = Expression.BINARY_COMPARISON
        elif (
            self.operator in ['+', '-', '*', '/']
            and len(self.children) >= 2
        ):
            if self.operator in ['-', '/']:
                assert len(self.children) == 2
            typ = Expression.BINARY_ARITHMETIC
        elif self.operator in ['IN', 'NOT IN']:
            typ = Expression.IN_OP
        elif self.operator in Expression.FUNC2TYPE:
            typ = Expression.FUNC2TYPE[self.operator]
        elif self.operator == 'IRI_FUNC':
            assert (
                self.children and isinstance(self.children[0], NodeTerm)
                and self.children[0].type & NodeTerm.IRI
            )
            typ = Expression.IRI_FUNC
        else:
            raise NotImplementedError

        super().__init__(lexstart, lexstop, typ)

    def to_str(self):
        def call_to_str():
            distinct = 'DISTINCT ' if self.is_distinct else ''
            args = ', '.join([str(x) for x in self.children])
            return f'{self.operator}({distinct}{args})'

        def call_group_concat():
            distinct = 'DISTINCT ' if self.is_distinct else ''
            sep = (
                f'; SEPARATOR={self.children[1]}'
                if len(self.children) > 1 else ''
            )
            return f'{self.operator}({distinct}{self.children[0]}{sep})' 
       
        def call_exists():
            return f'{self.operator} {self.children[0]}'

        call_handler = defaultdict(
            lambda: call_to_str,
            {
                'GROUP_CONCAT': call_group_concat,
                'EXISTS': call_exists,
                'NOT EXISTS': call_exists,
            }
        )
        if self.type & Expression.NOP:
            return f'({self.children[0]})'
        elif self.type & Expression.UNARY_OP:
            return f'{self.operator}{self.children[0]}'
        elif self.type & Expression.BINARY_OP:
            return f' {self.operator} '.join([str(e) for e in self.children])
        elif self.type & Expression.IN_OP:
            exprs = ', '.join([str(e) for e in self.children[1:]])
            return f'{self.children[0]} {self.operator} ({exprs})'
        elif self.type & Expression.BUILTIN_CALL:
            return call_handler[self.operator]()
        elif self.type & Expression.IRI_FUNC:
            args = ', '.join([str(x) for x in self.children[1:]])
            return f'{self.children[0]}({args})'
        else:
            raise NotImplementedError


#########################################################
#
#  GraphPattern
#
#########################################################

class GraphPattern(QueryComponent):
    '''
    Elements that can be part of a Group Graph Pattern.
    '''
    TRIPLES_BLOCK = gen_type()  # TriplesBlock
    GROUP         = gen_type()  # GroupGraphPattern
    UNION         = gen_type()  # UnionGraphPattern
    OPTIONAL      = gen_type()  # OptionalGraphPattern
    MINUS         = gen_type()  # MinusGraphPattern
    GRAPH         = gen_type()  # GraphGraphPattern
    SERVICE       = gen_type()  # ServiceGraphPattern
    FILTER        = gen_type()  # Filter
    BIND          = gen_type()  # Bind
    INLINE_DATA   = gen_type()  # InlineData
    SUB_SELECT    = gen_type()  # SubSelect
    
    TYPE = (
        TRIPLES_BLOCK | GROUP | UNION | OPTIONAL | MINUS | GRAPH
        | SERVICE | FILTER | BIND | INLINE_DATA | SUB_SELECT
    )
    
    TYPE2KEYWORD = {
        TRIPLES_BLOCK: None,
        GROUP: None,
        UNION: 'UNION',
        OPTIONAL: 'OPTIONAL',
        MINUS: 'MINUS',
        GRAPH: 'GRAPH',
        SERVICE: 'SERVICE',
        FILTER: 'FILTER',
        BIND: 'BIND',
        INLINE_DATA: 'VALUES',
        SUB_SELECT: None,
    }
    
    # check the number of children 
    CHILDREN_CHECKER = {
        TRIPLES_BLOCK: lambda x: len(x) >= 1,
        GROUP: lambda x: True,
        UNION: lambda x: len(x) >= 2,
        OPTIONAL: lambda x: len(x) == 1,
        MINUS: lambda x: len(x) == 1,
        GRAPH: lambda x: len(x) == 2,
        SERVICE: lambda x: len(x) == 2,
        FILTER: lambda x: len(x) == 1,
        BIND: lambda x: len(x) == 2,
        INLINE_DATA: lambda x: len(x) >= 1,
        SUB_SELECT: lambda x: len(x) == 1,
    }
    
    def __init__(
        self, lexstart: int, lexstop: int,
        children: Union[
            List[TriplesPath],              # TriplesBlock
            List[GraphPattern],             # Group, Union
            Tuple[GraphPattern],            # Optional, Minus
            Tuple[NodeTerm, GraphPattern],  # Graph, Service
            Tuple[Expression],              # Filter
            Tuple[Expression, NodeTerm],    # Bind          
            List[List[NodeTerm]],           # InlineData
            Tuple[Query],                   # SubSelect
        ],
        typ: int,
        is_silent: bool = False,
    ):
        self.children = children
        self.is_silent = is_silent
        
        if self.is_silent:
            assert typ & GraphPattern.SERVICE
        assert self.CHILDREN_CHECKER[typ](self.children)
        
        super().__init__(lexstart, lexstop, typ)

    def to_str(self):
        if self.type & GraphPattern.TRIPLES_BLOCK:
            return '.\n'.join([str(t) for t in self.children])
        elif self.type & GraphPattern.GROUP:
            return '{\n' + '\n'.join([str(g) for g in self.children]) + '\n}'
        elif self.type & GraphPattern.UNION:
            return ' UNION '.join([str(g) for g in self.children])
        elif self.type & (
            GraphPattern.OPTIONAL | GraphPattern.MINUS | GraphPattern.GRAPH
            | GraphPattern.SERVICE | GraphPattern.FILTER
        ):
            silent = 'SILENT ' if self.is_silent else ''
            children = ' '.join([str(x) for x in self.children])
            return f'{self.TYPE2KEYWORD[self.type]} {silent}{children}'
        elif self.type & GraphPattern.BIND:
            return f'BIND({self.children[0]} AS {self.children[1]})'
        elif self.type & GraphPattern.INLINE_DATA:
            vars = ' '.join([str(x) for x in self.children[0]])
            data_block = '\n'.join([
                '(' + ' '.join([str(x) for x in row]) + ')'
                for row in self.children[1:]
            ])
            return f'VALUES ({vars}) {{\n{data_block}\n}}'
        elif self.type & GraphPattern.SUB_SELECT:
            return '{\n' + str(self.children[0]) + '\n}'
        else:
            raise NotImplementedError


#########################################################
#
#  Query
#
#########################################################

# https://docs.python.org/3.6/library/typing.html
T = TypeVar('T')

class ComponentWrapper(QueryComponent, Generic[T]):
    '''
    Wrapper for a single value or a sequence of values.
    '''
    def __init__(self, lexstart: int, lexstop: int, value: T):
        if isinstance(value, (QueryComponent, str)):
            self.is_single = True
        elif isinstance(value, Sequence):
            self.is_single = False
        else:
            raise TypeError(
                f'{value} at position {lexstart} is not a valid value '
                'for ComponentWrapper.'
            )
        self.value = value
        QueryComponent.__init__(self, lexstart, lexstop, 0)

    def to_str(self) -> str:
        if self.is_single:
            return str(self.value)
        seq = ', '.join(str(x) for x in self.value)
        return f'{self.__class__.__name__}[{seq}]'

    def __eq__(self, other):
        if isinstance(other, ComponentWrapper):
            return self.value == other.value 
        return self.value == other

    def __hash__(self):
        if not self.is_single:
            return hash(tuple(self.value))
        return hash(self.value)

    def __getitem__(self, key):
        if self.is_single and isinstance(self.value, QueryComponent):
            return self.value
        return self.value[key]

    def __len__(self):
        if self.is_single and isinstance(self.value, QueryComponent):
            return 1
        return len(self.value)


class Query(QueryComponent):
    '''
    SPARQL Query
    '''
    SELECT     = gen_type()  # SelectQuery
    CONSTRUCT  = gen_type()  # ConstructQuery
    DESCRIBE   = gen_type()  # DescribeQuery
    ASK        = gen_type()  # AskQuery
    SUB_SELECT = gen_type()  # SubSelect
    
    TYPE = SELECT | CONSTRUCT | DESCRIBE | ASK | SUB_SELECT

    TYPE2KEYWORD = {
        SELECT: 'SELECT',
        CONSTRUCT: 'CONSTRUCT',
        DESCRIBE: 'DESCRIBE',
        ASK: 'ASK',
        SUB_SELECT: 'SELECT',
    }

    def __init__(
        self, lexstart: int, lexstop: int,
        typ: int,
        prologue: Optional[List[Union[
            ComponentWrapper[Tuple[str, NodeTerm]],
            ComponentWrapper[Tuple[str, NodeTerm, NodeTerm]],
        ]]] = None,
        select_modifier: Optional[ComponentWrapper[str]] = None,
        target: Optional[Union[
            List[Union[
                NodeTerm, ComponentWrapper[Tuple[Expression, NodeTerm]]
            ]],
            NodeTerm,      # '*', for SELECT and DESCRIBE
            GraphPattern,  # for CONSTRUCT
        ]] = None,
        dataset: Optional[List[Union[
            ComponentWrapper[Tuple[str, NodeTerm]],
            ComponentWrapper[Tuple[str, str, NodeTerm]],
        ]]] = None,
        pattern: Optional[GraphPattern] = None,
        group_by: Optional[ComponentWrapper[
            List[Union[
                NodeTerm, Expression,
                ComponentWrapper[Tuple[Expression, NodeTerm]],
            ]]
        ]] = None,
        having: Optional[ComponentWrapper[
            List[Expression]
        ]] = None,
        order_by: Optional[ComponentWrapper[
            List[Union[
                NodeTerm, Expression, 
                ComponentWrapper[Tuple[str, Expression]],
            ]]
        ]] = None,
        limit: Optional[ComponentWrapper[NodeTerm]] = None,
        offset: Optional[ComponentWrapper[NodeTerm]] = None,
        values: Optional[ComponentWrapper[
            List[List[NodeTerm]]
        ]] = None,
        raw_sparql: Optional[str] = None,
    ):
        self.prologue = prologue
        self.select_modifier = select_modifier
        self.target = target
        self.dataset = dataset
        self.pattern = pattern
        
        self.group_by = group_by
        self.having = having
        self.order_by = order_by
        self.limit = limit
        self.offset = offset
        self.values = values
        
        self.raw_sparql = raw_sparql

        if self.select_modifier is not None:
            assert typ & (Query.SELECT | Query.SUB_SELECT)
            assert self.select_modifier in ['DISTINCT', 'REDUCED']

        if typ & Query.SUB_SELECT:
            assert self.prologue is None
            assert self.dataset is None
            assert self.raw_sparql is None
        elif typ & Query.CONSTRUCT:
            if isinstance(self.target, GraphPattern):
                assert len(self.target.children) <= 1
                if self.target.children:
                    tb = self.target.children[0]
                    assert (
                        isinstance(tb, GraphPattern)
                        and tb.type & GraphPattern.TRIPLES_BLOCK
                    )
            else:
                assert self.target is None
        elif typ & Query.ASK:
            assert self.target is None

        super().__init__(lexstart, lexstop, typ)
        
    def to_str(self):
        res = ''
        if self.prologue is not None:
            res += '\n'.join([
                ' '.join([str(x) for x in row])
                for row in self.prologue
            ]) + '\n'
        res += self.TYPE2KEYWORD[self.type]
        if self.select_modifier is not None:
            res += ' ' + str(self.select_modifier)
        if self.target is not None:
            if isinstance(self.target, (NodeTerm, GraphPattern)):
                res += ' ' + str(self.target)
            elif isinstance(self.target, Sequence):
                for t in self.target:
                    if isinstance(t, NodeTerm):
                        res += ' ' + str(t)
                    elif isinstance(t, ComponentWrapper):
                        res += f' ({t[0]} AS {t[1]})'
                    else:
                        raise NotImplementedError
            else:
                raise NotImplementedError
        if self.dataset is not None: 
            res += '\n' + '\n'.join([
                ' '.join([str(x) for x in d])
                for d in self.dataset
            ])
        if self.pattern is not None:
            res += '\nWHERE ' + str(self.pattern)

        if self.group_by is not None:
            res += '\nGROUP BY ' + ' '.join([
                f'({x[0]} AS {x[1]})'
                if isinstance(x, ComponentWrapper) else str(x)
                for x in self.group_by
            ])
        if self.having is not None:
            res += '\nHAVING ' + ' '.join([str(x) for x in self.having])
        if self.order_by is not None:
            res += '\nORDER BY ' + ' '.join([
                x[0] + ' ' + str(x[1])
                if isinstance(x, ComponentWrapper) else str(x)
                for x in self.order_by
            ])
        if self.limit is not None:
            res += '\nLIMIT ' + str(self.limit)
        if self.offset is not None:
            res += '\nOFFSET ' + str(self.offset)
        if self.values is not None:
            vars = ' '.join([str(x) for x in self.values[0]])
            data_block = '\n'.join([
                '(' + ' '.join([str(x) for x in row]) + ')'
                for row in self.values[1:]
            ])
            res += f'\nVALUES ({vars}) {{\n{data_block}\n}}'
        return res

