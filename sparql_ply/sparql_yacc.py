'''
Copyright (c) 2024 qmj0923
https://github.com/qmj0923/SPARQL-PLY
'''

import os
import sys
from ply import lex, yacc

# https://stackoverflow.com/questions/16981921/relative-imports-in-python-3
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from sparql_ply import sparql_lex
from sparql_ply.sparql_lex import tokens
from sparql_ply.components import (
    NodeTerm, PropertyPath, CollectionPath, BlankNodePath, TriplesPath,
    GraphPattern, Expression, Query, QueryComponent, ComponentWrapper,
)


def p_error(t):
    if t is None:
        raise TypeError(f'Unknown text')
    else:
        raise TypeError(
            f'[Yacc] Unknown text {t.value} in line {t.lexer.lineno}'
            f' at position {t.lexpos} in the following text: \n'
            f'{t.lexer.lexdata}'
        )


def p_empty(p):
    '''
    empty :
    '''
    pass


############################################################
#  VarOrTerm
'''
[106] VarOrTerm ::= Var | GraphTerm
[108] Var ::= VAR1 | VAR2
[109] GraphTerm ::= iri | RDFLiteral | NumericLiteral | BooleanLiteral | BlankNode | NIL
[129] RDFLiteral ::= String ( LANGTAG | ( '^^' iri ) )?
[130] NumericLiteral ::= NumericLiteralUnsigned | NumericLiteralPositive | NumericLiteralNegative
[131] NumericLiteralUnsigned ::= INTEGER | DECIMAL | DOUBLE
[132] NumericLiteralPositive ::= INTEGER_POSITIVE | DECIMAL_POSITIVE | DOUBLE_POSITIVE
[133] NumericLiteralNegative ::= INTEGER_NEGATIVE | DECIMAL_NEGATIVE | DOUBLE_NEGATIVE
[134] BooleanLiteral ::= 'true' | 'false'
[135] String ::= STRING_LITERAL1 | STRING_LITERAL2 | STRING_LITERAL_LONG1 | STRING_LITERAL_LONG2
[136] iri ::= IRIREF | PrefixedName
[137] PrefixedName ::= PNAME_LN | PNAME_NS
[138] BlankNode ::= BLANK_NODE_LABEL | ANON
'''
############################################################

def rule_node_term(p, node_type, idx=1):
    lexstart = p.lexpos(idx)
    lexstop = lexstart + len(str(p[idx]))
    return NodeTerm(lexstart, lexstop, str(p[idx]), node_type)


# [106]
def p_var_or_term(p):
    '''
    var_or_term : var
                | graph_term
    '''
    p[0] = p[1]


# [108]
def p_var(p):
    '''
    var : VAR1
        | VAR2
    '''
    p[0] = rule_node_term(p, NodeTerm.VAR)


# [109]
def p_graph_term_0(p):
    '''
    graph_term : iri
               | rdf_literal
               | numeric_literal
               | boolean_literal
               | blank_node
    '''
    p[0] = p[1]


def p_graph_term_1(p):
    '''
    graph_term : NIL
    '''
    p[0] = rule_node_term(p, NodeTerm.NIL)


# [129]
def p_rdf_literal_0(p):
    '''
    rdf_literal : string
    '''
    lexstart, lexstop, content = p[1]
    p[0] = NodeTerm(
        lexstart, lexstop, content,
        NodeTerm.RDF_LITERAL,
    )

def p_rdf_literal_1(p):
    '''
    rdf_literal : string LANGTAG
    '''
    lexstart, _, content = p[1]
    lexstop = p.lexpos(2) + len(str(p[2]))
    p[0] = NodeTerm(
        lexstart, lexstop, content, 
        NodeTerm.RDF_LITERAL, language=str(p[2])[1:],
    )

def p_rdf_literal_2(p):
    '''
    rdf_literal : string DTYPE iri
    '''
    lexstart, _, content = p[1]
    lexstop = p[3].lexstop
    p[0] = NodeTerm(
        lexstart, lexstop, content,
        NodeTerm.RDF_LITERAL, datatype=p[3],
    )


# [135]
def p_string(p):
    '''
    string : STRING_LITERAL1
           | STRING_LITERAL2
           | STRING_LITERAL_LONG1
           | STRING_LITERAL_LONG2
    '''
    lexstart = p.lexpos(1)
    lexstop = lexstart + len(str(p[1]))
    p[0] = lexstart, lexstop, str(p[1])


# [130]
def p_numeric_literal(p):
    '''
    numeric_literal : numeric_literal_unsigned
                    | numeric_literal_positive
                    | numeric_literal_negative
    '''
    p[0] = p[1]


# [131]
def p_numeric_literal_unsigned_0(p):
    '''
    numeric_literal_unsigned : INTEGER
    '''
    p[0] = rule_node_term(p, NodeTerm.INTEGER)


def p_numeric_literal_unsigned_1(p):
    '''
    numeric_literal_unsigned : DECIMAL
    '''
    p[0] = rule_node_term(p, NodeTerm.DECIMAL)


def p_numeric_literal_unsigned_2(p):
    '''
    numeric_literal_unsigned : DOUBLE
    '''
    p[0] = rule_node_term(p, NodeTerm.DOUBLE)


# [132]
def p_numeric_literal_positive_0(p):
    '''
    numeric_literal_positive : INTEGER_POSITIVE
    '''
    p[0] = rule_node_term(p, NodeTerm.INTEGER)


def p_numeric_literal_positive_1(p):
    '''
    numeric_literal_positive : DECIMAL_POSITIVE
    '''
    p[0] = rule_node_term(p, NodeTerm.DECIMAL)


def p_numeric_literal_positive_2(p):
    '''
    numeric_literal_positive : DOUBLE_POSITIVE
    '''
    p[0] = rule_node_term(p, NodeTerm.DOUBLE)


# [133]
def p_numeric_literal_negative_0(p):
    '''
    numeric_literal_negative : INTEGER_NEGATIVE
    '''
    p[0] = rule_node_term(p, NodeTerm.INTEGER)


def p_numeric_literal_negative_1(p):
    '''
    numeric_literal_negative : DECIMAL_NEGATIVE
    '''
    p[0] = rule_node_term(p, NodeTerm.DECIMAL)


def p_numeric_literal_negative_2(p):
    '''
    numeric_literal_negative : DOUBLE_NEGATIVE
    '''
    p[0] = rule_node_term(p, NodeTerm.DOUBLE)


# [134]
def p_boolean_literal(p):
    '''
    boolean_literal : TRUE
                    | FALSE
    '''
    p[0] = rule_node_term(p, NodeTerm.BOOLEAN)


# [136]
def p_iri_0(p):
    '''
    iri : IRIREF
    '''
    p[0] = rule_node_term(p, NodeTerm.IRIREF)


def p_iri_1(p):
    '''
    iri : PNAME_LN
        | PNAME_NS
    '''
    p[0] = rule_node_term(p, NodeTerm.PREFIXED_NAME)


# [138]
def p_blank_node_0(p):
    '''
    blank_node : BLANK_NODE_LABEL
    '''
    p[0] = rule_node_term(p, NodeTerm.BLANK_NODE_LABEL)


def p_blank_node_1(p):
    '''
    blank_node : ANON
    '''
    p[0] = rule_node_term(p, NodeTerm.ANON)


############################################################
#  Path
'''
[88] Path ::= PathAlternative
[89] PathAlternative ::= PathSequence ( '|' PathSequence )*
[90] PathSequence ::= PathEltOrInverse ( '/' PathEltOrInverse )*
[91] PathElt ::= PathPrimary PathMod?
[92] PathEltOrInverse ::= PathElt | '^' PathElt
[93] PathMod ::= '?' | '*' | '+'
[94] PathPrimary ::= iri | 'a' | '!' PathNegatedPropertySet | '(' Path ')'
[95] PathNegatedPropertySet ::= PathOneInPropertySet | '(' ( PathOneInPropertySet ( '|' PathOneInPropertySet )* )? ')'
[96] PathOneInPropertySet ::= iri | 'a' | '^' ( iri | 'a' )
'''
############################################################

# [88]
def p_path(p):
    '''
    path : path_alternative
    '''
    p[0] = p[1]


# [89]
def p_path_alternative(p):
    '''
    path_alternative : path_sequence path_alternative_more
    '''
    if not p[2]:
        p[0] = p[1]
        return

    operands = [p[1]] + p[2]
    lexstart = operands[0].lexstart
    lexstop = operands[-1].lexstop
    p[0] =  PropertyPath(lexstart, lexstop, operands, '|')


def p_path_alternative_more_0(p):
    '''
    path_alternative_more : empty
    '''
    p[0] = []


def p_path_alternative_more_1(p):
    '''
    path_alternative_more : VBAR path_sequence path_alternative_more
    '''
    p[0] = [p[2]] + p[3]


# [90]
def p_path_sequence(p):
    '''
    path_sequence : path_elt_or_inverse path_sequence_more
    '''
    if not p[2]:
        p[0] = p[1]
        return
    
    operands = [p[1]] + p[2]
    lexstart = operands[0].lexstart
    lexstop = operands[-1].lexstop
    p[0] = PropertyPath(lexstart, lexstop, operands, '/')


def p_path_sequence_more_0(p):
    '''
    path_sequence_more : empty
    '''
    p[0] = []


def p_path_sequence_more_1(p):
    '''
    path_sequence_more : SLASH path_elt_or_inverse path_sequence_more
    '''
    p[0] = [p[2]] + p[3]


# [92]
def p_path_elt_or_inverse_0(p):
    '''
    path_elt_or_inverse : path_elt
    '''
    p[0] = p[1]


def p_path_elt_or_inverse_1(p):
    '''
    path_elt_or_inverse : CARAT path_elt
    '''
    lexstart = p.lexpos(1)
    lexstop = p[2].lexstop
    p[0] = PropertyPath(lexstart, lexstop, [p[2]], str(p[1]))


# [91]
def p_path_elt_0(p):
    '''
    path_elt : path_primary
    '''
    p[0] = p[1]


def p_path_elt_1(p):
    '''
    path_elt : path_primary path_mod
    '''
    lexstart = p[1].lexstart
    _, lexstop, mod = p[2]
    p[0] = PropertyPath(lexstart, lexstop, [p[1]], mod)


# [93]
def p_path_mod(p):
    '''
    path_mod : QMARK
             | STAR
             | SC_PLUS
    '''
    lexstart = p.lexpos(1)
    lexstop = lexstart + len(str(p[1]))
    p[0] = lexstart, lexstop, str(p[1])


# [94]
def p_path_primary_0(p):
    '''
    path_primary : iri
    '''
    p[0] = p[1]


def p_path_primary_1(p):
    '''
    path_primary : KW_A
    '''
    p[0] = rule_node_term(p, NodeTerm.SPECIAL)


def p_path_primary_2(p):
    '''
    path_primary : BANG path_negated_property_set
    '''
    lexstart = p.lexpos(1)
    lexstop = p[2].lexstop
    p[0] = PropertyPath(lexstart, lexstop, [p[2]], '!')


def p_path_primary_3(p):
    '''
    path_primary : LPAREN path RPAREN
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(3) + len(str(p[3]))
    p[0] = PropertyPath(lexstart, lexstop, [p[2]])


# [95]
def p_path_negated_property_set_0(p):
    '''
    path_negated_property_set : path_one_in_property_set
    '''
    p[0] = p[1]


def p_path_negated_property_set_1(p):
    '''
    path_negated_property_set : LPAREN path_one_in_property_set path_negated_property_set_more RPAREN
    '''
    # Although `!()` is allowed in the grammar, it is not valid in the parser.
    # Because `!()` will be lexed as `BANG NIL`, not `BANG LPAREN RPAREN`.
    if not p[3]:
        inner_pp = p[2]
    else:
        operands = [p[2]] + p[3]
        lexstart = operands[0].lexstart
        lexstop = operands[-1].lexstop
        inner_pp = PropertyPath(lexstart, lexstop, operands, '|')

    lexstart = p.lexpos(1)
    lexstop = p.lexpos(4) + len(str(p[4]))
    p[0] = PropertyPath(lexstart, lexstop, [inner_pp])


def p_path_negated_property_set_more_0(p):
    '''
    path_negated_property_set_more : empty
    '''
    p[0] = []


def p_path_negated_property_set_more_1(p):
    '''
    path_negated_property_set_more : VBAR path_one_in_property_set path_negated_property_set_more
    '''
    p[0] = [p[2]] + p[3]


# [96]
def p_path_one_in_property_set_0(p):
    '''
    path_one_in_property_set : iri
    '''
    p[0] = p[1]


def p_path_one_in_property_set_1(p):
    '''
    path_one_in_property_set : KW_A
    '''
    p[0] = rule_node_term(p, NodeTerm.SPECIAL)


def p_path_one_in_property_set_2(p):
    '''
    path_one_in_property_set : CARAT iri
    '''
    lexstart = p.lexpos(1)
    lexstop = p[2].lexstop
    p[0] = PropertyPath(lexstart, lexstop, [p[2]], '^')


def p_path_one_in_property_set_3(p):
    '''
    path_one_in_property_set : CARAT KW_A
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(2) + len(str(p[2]))
    node = rule_node_term(p, NodeTerm.SPECIAL, 2)
    p[0] = PropertyPath(lexstart, lexstop, [node], '^')


############################################################
#  TriplesSameSubjectPath
'''
[81] TriplesSameSubjectPath ::= VarOrTerm PropertyListPathNotEmpty | TriplesNodePath PropertyListPath
[82] PropertyListPath ::= PropertyListPathNotEmpty?
[83] PropertyListPathNotEmpty ::= ( VerbPath | VerbSimple ) ObjectListPath ( ';' ( ( VerbPath | VerbSimple ) ObjectList )? )*
[84] VerbPath ::= Path
[85] VerbSimple ::= Var

[86] ObjectListPath ::= ObjectPath ( ',' ObjectPath )*
[87] ObjectPath ::= GraphNodePath
[105] GraphNodePath ::= VarOrTerm | TriplesNodePath
[100] TriplesNodePath ::= CollectionPath | BlankNodePropertyListPath
[103] CollectionPath ::= '(' GraphNodePath+ ')'
[101] BlankNodePropertyListPath ::= '[' PropertyListPathNotEmpty ']'

[79] ObjectList ::= Object ( ',' Object )*
[80] Object ::= GraphNode
[104] GraphNode ::= VarOrTerm | TriplesNode
[98] TriplesNode ::= Collection | BlankNodePropertyList
[102] Collection ::= '(' GraphNode+ ')'
[99] BlankNodePropertyList ::= '[' PropertyListNotEmpty ']'

[77] PropertyListNotEmpty ::= Verb ObjectList ( ';' ( Verb ObjectList )? )*
[78] Verb ::= VarOrIri | 'a'
[107] VarOrIri ::= Var | iri
'''
############################################################

# [81]
def p_triples_same_subject_path(p):
    '''
    triples_same_subject_path : var_or_term property_list_path_not_empty
                              | triples_node_path property_list_path
    '''
    lexstart = p[1].lexstart
    lexstop = (
        p[2][-1][1][-1].lexstop
        if p[2] is not None else p[1].lexstop
    )
    p[0] = TriplesPath(lexstart, lexstop, p[1], p[2])


# [82]
def p_property_list_path_0(p):
    '''
    property_list_path : empty
    '''
    p[0] = None


def p_property_list_path_1(p):
    '''
    property_list_path : property_list_path_not_empty
    '''
    p[0] = p[1]


# [83]
def p_property_list_path_not_empty(p):
    '''
    property_list_path_not_empty : verb_path object_list_path property_list_path_not_empty_more
                                 | verb_simple object_list_path property_list_path_not_empty_more
    '''
    p[0] = [[p[1], p[2]]] + p[3]


def p_property_list_path_not_empty_more_0(p):
    '''
    property_list_path_not_empty_more :  empty
    '''
    p[0] = []


def p_property_list_path_not_empty_more_1(p):
    '''
    property_list_path_not_empty_more : SEMICOLON property_list_path_not_empty_more
    '''
    p[0] = p[2]


def p_property_list_path_not_empty_more_2(p):
    '''
    property_list_path_not_empty_more : SEMICOLON verb_path object_list property_list_path_not_empty_more
                                      | SEMICOLON verb_simple object_list property_list_path_not_empty_more
    '''
    p[0] = [[p[2], p[3]]] + p[4]


# [84]
def p_verb_path(p):
    '''
    verb_path : path
    '''
    p[0] = p[1]


# [85]
def p_verb_simple(p):
    '''
    verb_simple : var
    '''
    p[0] = p[1]


########## ObjectListPath ##########

# [86]
def p_object_list_path(p):
    '''
    object_list_path : object_path object_list_path_more
    '''
    p[0] = [p[1]] + p[2]


def p_object_list_path_more_0(p):
    '''
    object_list_path_more : empty
    '''
    p[0] = []


def p_object_list_path_more_1(p):
    '''
    object_list_path_more : COMMA object_path object_list_path_more
    '''
    p[0] = [p[2]] + p[3]


# [87]
def p_object_path(p):
    '''
    object_path : graph_node_path
    '''
    p[0] = p[1]


# [105]
def p_graph_node_path(p):
    '''
    graph_node_path : var_or_term
                    | triples_node_path
    '''
    p[0] = p[1]


# [100]
def p_triples_node_path(p):
    '''
    triples_node_path : collection_path
                      | blank_node_property_list_path
    '''
    p[0] = p[1]


# [103]
def p_collection_path(p):
    '''
    collection_path : LPAREN graph_node_path graph_nodes_path RPAREN
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(4) + len(str(p[4]))
    p[0] = CollectionPath(lexstart, lexstop, [p[2]] + p[3])


def p_graph_nodes_path_0(p):
    '''
    graph_nodes_path : empty
    '''
    p[0] = []


def p_graph_nodes_path_1(p):
    '''
    graph_nodes_path : graph_node_path graph_nodes_path
    '''
    p[0] = [p[1]] + p[2]


# [101]
def p_blank_node_property_list_path(p):
    '''
    blank_node_property_list_path : LBRACKET property_list_path_not_empty RBRACKET
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(3) + len(str(p[3]))
    p[0] = BlankNodePath(lexstart, lexstop, p[2])


########## ObjectPath ##########

# [79]
def p_object_list(p):
    '''
    object_list : object object_list_more
    '''
    p[0] = [p[1]] + p[2]


def p_object_list_more_0(p):
    '''
    object_list_more : empty
    '''
    p[0] = []


def p_object_list_more_1(p):
    '''
    object_list_more : COMMA object object_list_more
    '''
    p[0] = [p[2]] + p[3]


# [80]
def p_object(p):
    '''
    object : graph_node
    '''
    p[0] = p[1]


# [104]
def p_graph_node(p):
    '''
    graph_node : var_or_term
               | triples_node
    '''
    p[0] = p[1]


# [98]
def p_triples_node(p):
    '''
    triples_node : collection
                 | blank_node_property_list
    '''
    p[0] = p[1]


# [102]
def p_collection(p):
    '''
    collection : LPAREN graph_node graph_nodes RPAREN
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(4) + len(str(p[4]))
    p[0] = CollectionPath(lexstart, lexstop, [p[2]] + p[3])


def p_graph_nodes_0(p):
    '''
    graph_nodes : empty
    '''
    p[0] = []


def p_graph_nodes_1(p):
    '''
    graph_nodes : graph_node graph_nodes
    '''
    p[0] = [p[1]] + p[2]


# [99]
def p_blank_node_property_list(p):
    '''
    blank_node_property_list : LBRACKET property_list_not_empty RBRACKET
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(3) + len(str(p[3]))
    p[0] = BlankNodePath(lexstart, lexstop, p[2])


########## PropertyListNotEmpty ##########

# [77]
def p_property_list_not_empty(p):
    '''
    property_list_not_empty : verb object_list property_list_not_empty_more
    '''
    p[0] = [[p[1], p[2]]] + p[3]


def p_property_list_not_empty_more_0(p):
    '''
    property_list_not_empty_more : empty
    '''
    p[0] = []


def p_property_list_not_empty_more_1(p):
    '''
    property_list_not_empty_more : SEMICOLON property_list_not_empty_more
    '''
    p[0] = p[2]


def p_property_list_not_empty_more_2(p):
    '''
    property_list_not_empty_more : SEMICOLON verb object_list property_list_not_empty_more
    '''
    p[0] = [[p[2], p[3]]] + p[4]


# [78]
def p_verb_0(p):
    '''
    verb : var_or_iri
    '''
    p[0] = p[1]


def p_verb_1(p):
    '''
    verb : KW_A
    '''
    p[0] = rule_node_term(p, NodeTerm.SPECIAL)


# [107]
def p_var_or_iri(p):
    '''
    var_or_iri : var
               | iri
    '''
    p[0] = p[1]


############################################################
#  Expression
'''
[71] ArgList ::= NIL | '(' 'DISTINCT'? Expression ( ',' Expression )* ')'
[72] ExpressionList ::= NIL | '(' Expression ( ',' Expression )* ')'
[110] Expression ::= ConditionalOrExpression
[111] ConditionalOrExpression ::= ConditionalAndExpression ( '||' ConditionalAndExpression )*
[112] ConditionalAndExpression ::= ValueLogical ( '&&' ValueLogical )*
[113] ValueLogical ::= RelationalExpression
[114] RelationalExpression ::= NumericExpression ( '=' NumericExpression | '!=' NumericExpression | '<' NumericExpression | '>' NumericExpression | '<=' NumericExpression | '>=' NumericExpression | 'IN' ExpressionList | 'NOT' 'IN' ExpressionList )?
[115] NumericExpression ::= AdditiveExpression
[116] AdditiveExpression ::= MultiplicativeExpression ( '+' MultiplicativeExpression | '-' MultiplicativeExpression | ( NumericLiteralPositive | NumericLiteralNegative ) ( ( '*' UnaryExpression ) | ( '/' UnaryExpression ) )* )*
[117] MultiplicativeExpression ::= UnaryExpression ( '*' UnaryExpression | '/' UnaryExpression )*
[118] UnaryExpression ::= '!' PrimaryExpression
                          | '+' PrimaryExpression
                          | '-' PrimaryExpression
                          | PrimaryExpression
[119] PrimaryExpression ::= BrackettedExpression | BuiltInCall | iriOrFunction | RDFLiteral | NumericLiteral | BooleanLiteral | Var
[120] BrackettedExpression ::= '(' Expression ')'
[121] BuiltInCall ::= Aggregate
                      | 'STR' '(' Expression ')'
                      | 'LANG' '(' Expression ')'
                      | 'LANGMATCHES' '(' Expression ',' Expression ')'
                      | 'DATATYPE' '(' Expression ')'
                      | 'BOUND' '(' Var ')'
                      | 'IRI' '(' Expression ')'
                      | 'URI' '(' Expression ')'
                      | 'BNODE' ( '(' Expression ')' | NIL )
                      | 'RAND' NIL
                      | 'ABS' '(' Expression ')'
                      | 'CEIL' '(' Expression ')'
                      | 'FLOOR' '(' Expression ')'
                      | 'ROUND' '(' Expression ')'
                      | 'CONCAT' ExpressionList
                      | SubstringExpression
                      | 'STRLEN' '(' Expression ')'
                      | StrReplaceExpression
                      | 'UCASE' '(' Expression ')'
                      | 'LCASE' '(' Expression ')'
                      | 'ENCODE_FOR_URI' '(' Expression ')'
                      | 'CONTAINS' '(' Expression ',' Expression ')'
                      | 'STRSTARTS' '(' Expression ',' Expression ')'
                      | 'STRENDS' '(' Expression ',' Expression ')'
                      | 'STRBEFORE' '(' Expression ',' Expression ')'
                      | 'STRAFTER' '(' Expression ',' Expression ')'
                      | 'YEAR' '(' Expression ')'
                      | 'MONTH' '(' Expression ')'
                      | 'DAY' '(' Expression ')'
                      | 'HOURS' '(' Expression ')'
                      | 'MINUTES' '(' Expression ')'
                      | 'SECONDS' '(' Expression ')'
                      | 'TIMEZONE' '(' Expression ')'
                      | 'TZ' '(' Expression ')'
                      | 'NOW' NIL
                      | 'UUID' NIL
                      | 'STRUUID' NIL
                      | 'MD5' '(' Expression ')'
                      | 'SHA1' '(' Expression ')'
                      | 'SHA256' '(' Expression ')'
                      | 'SHA384' '(' Expression ')'
                      | 'SHA512' '(' Expression ')'
                      | 'COALESCE' ExpressionList
                      | 'IF' '(' Expression ',' Expression ',' Expression ')'
                      | 'STRLANG' '(' Expression ',' Expression ')'
                      | 'STRDT' '(' Expression ',' Expression ')'
                      | 'sameTerm' '(' Expression ',' Expression ')'
                      | 'isIRI' '(' Expression ')'
                      | 'isURI' '(' Expression ')'
                      | 'isBLANK' '(' Expression ')'
                      | 'isLITERAL' '(' Expression ')'
                      | 'isNUMERIC' '(' Expression ')'
                      | RegexExpression
                      | ExistsFunc
                      | NotExistsFunc
[122] RegexExpression ::= 'REGEX' '(' Expression ',' Expression ( ',' Expression )? ')'
[123] SubstringExpression ::= 'SUBSTR' '(' Expression ',' Expression ( ',' Expression )? ')'
[124] StrReplaceExpression ::= 'REPLACE' '(' Expression ',' Expression ',' Expression ( ',' Expression )? ')'
[125] ExistsFunc ::= 'EXISTS' GroupGraphPattern
[126] NotExistsFunc ::= 'NOT' 'EXISTS' GroupGraphPattern
[127] Aggregate ::= 'COUNT' '(' 'DISTINCT'? ( '*' | Expression ) ')'
                    | 'SUM' '(' 'DISTINCT'? Expression ')'
                    | 'MIN' '(' 'DISTINCT'? Expression ')'
                    | 'MAX' '(' 'DISTINCT'? Expression ')'
                    | 'AVG' '(' 'DISTINCT'? Expression ')'
                    | 'SAMPLE' '(' 'DISTINCT'? Expression ')'
                    | 'GROUP_CONCAT' '(' 'DISTINCT'? Expression ( ';' 'SEPARATOR' '=' String )? ')'
[128] iriOrFunction ::= iri ArgList?
'''
############################################################

# [71]
def p_arg_list_0(p):
    '''
    arg_list : NIL
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(1) + len(str(p[1]))
    p[0] = lexstart, lexstop, [], False


def p_arg_list_1(p): 
    '''
    arg_list : LPAREN expression expression_list_more RPAREN
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(4) + len(str(p[4]))
    p[0] = lexstart, lexstop, [p[2]] + p[3], False


def p_arg_list_2(p): 
    '''
    arg_list : LPAREN DISTINCT expression expression_list_more RPAREN
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(5) + len(str(p[5]))
    p[0] = lexstart, lexstop, [p[3]] + p[4], True


def p_expression_list_more_0(p):
    '''
    expression_list_more : empty
    '''
    p[0] = []


def p_expression_list_more_1(p):
    '''
    expression_list_more : COMMA expression expression_list_more
    '''
    p[0] = [p[2]] + p[3]


# [72]
def p_expression_list_0(p):
    '''
    expression_list : NIL
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(1) + len(str(p[1]))
    p[0] = lexstart, lexstop, []


def p_expression_list_1(p):
    '''
    expression_list : LPAREN expression expression_list_more RPAREN
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(4) + len(str(p[4]))
    p[0] = lexstart, lexstop, [p[2]] + p[3]


# [110]
def p_expression(p):
    '''
    expression : conditional_or_expression
    '''
    p[0] = p[1]


########## Logical ##########

# [111]
def p_conditional_or_expression(p):     
    '''
    conditional_or_expression : conditional_and_expression conditional_or_expression_more
    '''
    if not p[2]:
        p[0] = p[1]
        return

    operands = [p[1]] + p[2]
    lexstart = operands[0].lexstart
    lexstop = operands[-1].lexstop
    p[0] = Expression(lexstart, lexstop, operands, '||')


def p_conditional_or_expression_more_0(p):     
    '''
    conditional_or_expression_more : empty
    '''
    p[0] = []


def p_conditional_or_expression_more_1(p):     
    '''
    conditional_or_expression_more : SC_OR conditional_and_expression conditional_or_expression_more
    '''
    p[0] = [p[2]] + p[3]


# [112]
def p_conditional_and_expression(p):     
    '''
    conditional_and_expression : value_logical conditional_and_expression_more
    '''
    if not p[2]:
        p[0] = p[1]
        return

    operands = [p[1]] + p[2]
    lexstart = operands[0].lexstart
    lexstop = operands[-1].lexstop
    p[0] = Expression(lexstart, lexstop, operands, '&&')


def p_conditional_and_expression_more_0(p):
    '''
    conditional_and_expression_more : empty
    '''
    p[0] = []


def p_conditional_and_expression_more_1(p):
    '''
    conditional_and_expression_more : SC_AND value_logical conditional_and_expression_more
    '''
    p[0] = [p[2]] + p[3]


# [113]
def p_value_logical(p):
    '''
    value_logical : relational_expression
    '''
    p[0] = p[1]


########## Comparison ##########

# [114]
def p_relational_expression_0(p):
    '''
    relational_expression : numeric_expression
    '''
    p[0] = p[1]


def p_relational_expression_1(p):
    '''
    relational_expression : numeric_expression EQ numeric_expression
                          | numeric_expression NE numeric_expression
                          | numeric_expression LT numeric_expression
                          | numeric_expression GT numeric_expression
                          | numeric_expression LE numeric_expression
                          | numeric_expression GE numeric_expression
    '''
    lexstart = p[1].lexstart
    lexstop = p[3].lexstop
    p[0] = Expression(lexstart, lexstop, [p[1], p[3]], str(p[2]))


def p_relational_expression_2(p):
    '''
    relational_expression : numeric_expression IN expression_list
    '''
    lexstart = p[1].lexstart
    _, lexstop, expr_list = p[3]
    p[0] = Expression(lexstart, lexstop, [p[1]] + expr_list, 'IN')
    

def p_relational_expression_3(p):
    '''
    relational_expression : numeric_expression NOT IN expression_list
    '''
    lexstart = p[1].lexstart
    _, lexstop, expr_list = p[4]
    p[0] = Expression(lexstart, lexstop, [p[1]] + expr_list, 'NOT IN')


########## Arithmetic ##########

# [115]
def p_numeric_expression(p):
    '''
    numeric_expression : additive_expression
    '''
    p[0] = p[1]


def create_arithmetic_expr(expr, expr_more, op1='+', op2='-'):
    '''
    expr: Expression
    expr_more: [[op, Expression], [op, Expression], ...]
    '''
    def stack_to_expr(st):
        if len(st) == 1:
            return st[0]
        lexstart = st[0].lexstart
        lexstop = st[-1].lexstop
        return Expression(lexstart, lexstop, st, op1)

    stack = [expr]
    for op, right in expr_more:
        if op == op1:
            stack.append(right)
        elif op == op2:
            left = stack_to_expr(stack)
            e = Expression(
                left.lexstart, right.lexstop, [left, right], op2
            )
            stack = [e]
        else:
            raise NotImplementedError

    return stack_to_expr(stack)


# [116]
def p_additive_expression(p):
    '''
    additive_expression : multiplicative_expression additive_expression_more
    '''
    p[0] = create_arithmetic_expr(p[1], p[2], '+', '-')


def p_additive_expression_more_0(p):
    '''
    additive_expression_more : empty
    '''
    p[0] = []


def p_additive_expression_more_1(p):
    '''
    additive_expression_more : SC_PLUS multiplicative_expression additive_expression_more
                             | SC_MINUS multiplicative_expression additive_expression_more
    '''
    p[0] = [[str(p[1]), p[2]]] + p[3]


def p_additive_expression_more_2(p):
    '''
    additive_expression_more : numeric_literal_positive multiplicative_expression_more additive_expression_more
                             | numeric_literal_negative multiplicative_expression_more additive_expression_more
    '''
    # '1 +2' will be lexed as ['1', '+2'], so we need to parse it as ['1', '+', '2']
    sign, value = p[1].value[:1], p[1].value[1:]
    nl = NodeTerm(p[1].lexstart + 1, p[1].lexstop, value, p[1].type)
    mult_expr = create_arithmetic_expr(nl, p[2], '*', '/')
    p[0] = [[sign, mult_expr]] + p[3]


def p_multiplicative_expression_more_0(p):
    '''
    multiplicative_expression_more : empty
    '''
    p[0] = []


def p_multiplicative_expression_more_1(p):
    '''
    multiplicative_expression_more : STAR unary_expression multiplicative_expression_more
                                   | SLASH unary_expression multiplicative_expression_more
    '''
    p[0] = [[str(p[1]), p[2]]] + p[3]


# [117]
def p_multiplicative_expression(p):
    '''
    multiplicative_expression : unary_expression multiplicative_expression_more
    '''
    p[0] = create_arithmetic_expr(p[1], p[2], '*', '/')
        

# [118]
def p_unary_expressio_0(p):
    '''
    unary_expression : primary_expression
    '''
    p[0] = p[1]


def p_unary_expression_1(p):
    '''
    unary_expression : BANG primary_expression
                     | SC_PLUS primary_expression
                     | SC_MINUS primary_expression
    '''
    lexstart = p.lexpos(1)
    lexstop = p[2].lexstop
    p[0] = Expression(lexstart, lexstop, [p[2]], str(p[1]))


# [119]
def p_primary_expression(p):
    '''
    primary_expression : bracketted_expression
                       | built_in_call
                       | iri_or_function
                       | rdf_literal
                       | numeric_literal
                       | boolean_literal
                       | var
    '''
    p[0] = p[1]


# [120]
def p_bracketted_expression(p):
    '''
    bracketted_expression : LPAREN expression RPAREN
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(3) + len(str(p[3]))
    p[0] = Expression(lexstart, lexstop, [p[2]])


########## Builtin Call ##########

def rule_call_expr(p, argc):
    idx_end = 2 + 2 * argc
    args = [p[i] for i in range(3, idx_end, 2)]
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(idx_end) + len(str(p[idx_end]))
    return Expression(lexstart, lexstop, args, str(p[1]))


# [121]
def p_built_in_call_0(p):
    '''
    built_in_call : BNODE NIL
                  | RAND NIL
                  | NOW NIL
                  | UUID NIL
                  | STRUUID NIL
    '''
    p[0] = rule_call_expr(p, 0)


def p_built_in_call_1(p):
    '''
    built_in_call : STR LPAREN expression RPAREN
                  | LANG LPAREN expression RPAREN
                  | DATATYPE LPAREN expression RPAREN
                  | BOUND LPAREN var RPAREN
                  | IRI LPAREN expression RPAREN
                  | URI LPAREN expression RPAREN
                  | BNODE LPAREN expression RPAREN
                  | ABS LPAREN expression RPAREN
                  | CEIL LPAREN expression RPAREN
                  | FLOOR LPAREN expression RPAREN
                  | ROUND LPAREN expression RPAREN
                  | STRLEN LPAREN expression RPAREN
                  | UCASE LPAREN expression RPAREN
                  | LCASE LPAREN expression RPAREN
                  | ENCODE_FOR_URI LPAREN expression RPAREN
                  | YEAR LPAREN expression RPAREN
                  | MONTH LPAREN expression RPAREN
                  | DAY LPAREN expression RPAREN
                  | HOURS LPAREN expression RPAREN
                  | MINUTES LPAREN expression RPAREN
                  | SECONDS LPAREN expression RPAREN
                  | TIMEZONE LPAREN expression RPAREN
                  | TZ LPAREN expression RPAREN
                  | MD5 LPAREN expression RPAREN
                  | SHA1 LPAREN expression RPAREN
                  | SHA256 LPAREN expression RPAREN
                  | SHA384 LPAREN expression RPAREN
                  | SHA512 LPAREN expression RPAREN
                  | ISIRI LPAREN expression RPAREN
                  | ISURI LPAREN expression RPAREN
                  | ISBLANK LPAREN expression RPAREN
                  | ISLITERAL LPAREN expression RPAREN
                  | ISNUMERIC LPAREN expression RPAREN
    '''
    p[0] = rule_call_expr(p, 1)


def p_built_in_call_2(p):
    '''
    built_in_call : LANGMATCHES LPAREN expression COMMA expression RPAREN
                  | CONTAINS LPAREN expression COMMA expression RPAREN
                  | STRSTARTS LPAREN expression COMMA expression RPAREN
                  | STRENDS LPAREN expression COMMA expression RPAREN
                  | STRBEFORE LPAREN expression COMMA expression RPAREN
                  | STRAFTER LPAREN expression COMMA expression RPAREN
                  | STRLANG LPAREN expression COMMA expression RPAREN
                  | STRDT LPAREN expression COMMA expression RPAREN
                  | SAMETERM LPAREN expression COMMA expression RPAREN
    '''
    p[0] = rule_call_expr(p, 2)


def p_built_in_call_3(p):
    '''
    built_in_call : IF LPAREN expression COMMA expression COMMA expression RPAREN
    '''
    p[0] = rule_call_expr(p, 3)


def p_built_in_call_4(p):
    '''
    built_in_call : CONCAT expression_list
                  | COALESCE expression_list
    '''
    lexstart = p.lexpos(1)
    _, lexstop, expr_list = p[2]
    p[0] = Expression(lexstart, lexstop, expr_list, str(p[1]))


def p_built_in_call_5(p):
    '''
    built_in_call : aggregate
                  | substring_expression
                  | str_replace_expression
                  | regex_expression
                  | exists_func
                  | not_exists_func
    '''
    p[0] = p[1]


# [122]
def p_regex_expression_0(p):
    '''
    regex_expression : REGEX LPAREN expression COMMA expression RPAREN
    '''
    p[0] = rule_call_expr(p, 2)


def p_regex_expression_1(p):
    '''
    regex_expression : REGEX LPAREN expression COMMA expression COMMA expression RPAREN
    '''
    p[0] = rule_call_expr(p, 3)


# [123]
def p_substring_expression_0(p):
    '''
    substring_expression : SUBSTR LPAREN expression COMMA expression RPAREN
    '''
    p[0] = rule_call_expr(p, 2)


def p_substring_expression_1(p):
    '''
    substring_expression : SUBSTR LPAREN expression COMMA expression COMMA expression RPAREN
    '''
    p[0] = rule_call_expr(p, 3)


# [124]
def p_str_replace_expression_0(p):
    '''
    str_replace_expression : REPLACE LPAREN expression COMMA expression COMMA expression RPAREN
    '''
    p[0] = rule_call_expr(p, 3)


def p_str_replace_expression_1(p):
    '''
    str_replace_expression : REPLACE LPAREN expression COMMA expression COMMA expression COMMA expression RPAREN
    '''
    p[0] = rule_call_expr(p, 4)


# [125]
def p_exists_func(p):
    '''
    exists_func : EXISTS group_graph_pattern
    '''
    lexstart = p.lexpos(1)
    lexstop = p[2].lexstop
    p[0] = Expression(lexstart, lexstop, [p[2]], 'EXISTS')


# [126]
def p_not_exists_func(p):
    '''
    not_exists_func : NOT EXISTS group_graph_pattern
    '''
    lexstart = p.lexpos(1)
    lexstop = p[3].lexstop
    p[0] = Expression(lexstart, lexstop, [p[3]], 'NOT EXISTS')


# [127]
def p_aggregate_0(p):
    '''
    aggregate : COUNT LPAREN STAR RPAREN
    '''
    node_star = rule_node_term(p, NodeTerm.SPECIAL, 3)
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(4) + len(str(p[4]))
    p[0] = Expression(lexstart, lexstop, [node_star], str(p[1]))


def p_aggregate_1(p):
    '''
    aggregate : COUNT LPAREN DISTINCT STAR RPAREN
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(5) + len(str(p[5]))
    node_star = rule_node_term(p, NodeTerm.SPECIAL, 4)
    p[0] = Expression(lexstart, lexstop, [node_star], str(p[1]), True)


def p_aggregate_2(p):
    '''
    aggregate : COUNT LPAREN expression RPAREN
              | SUM LPAREN expression RPAREN
              | MIN LPAREN expression RPAREN
              | MAX LPAREN expression RPAREN
              | AVG LPAREN expression RPAREN
              | SAMPLE LPAREN expression RPAREN
              | GROUP_CONCAT LPAREN expression RPAREN
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(4) + len(str(p[4]))
    p[0] = Expression(lexstart, lexstop, [p[3]], str(p[1]))


def p_aggregate_3(p):
    '''
    aggregate : COUNT LPAREN DISTINCT expression RPAREN
              | SUM LPAREN DISTINCT expression RPAREN
              | MIN LPAREN DISTINCT expression RPAREN
              | MAX LPAREN DISTINCT expression RPAREN
              | AVG LPAREN DISTINCT expression RPAREN
              | SAMPLE LPAREN DISTINCT expression RPAREN
              | GROUP_CONCAT LPAREN DISTINCT expression RPAREN
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(5) + len(str(p[5]))
    p[0] = Expression(lexstart, lexstop, [p[4]], str(p[1]), True)


def p_aggregate_4(p):
    '''
    aggregate : GROUP_CONCAT LPAREN expression SEMICOLON SEPARATOR EQ string RPAREN
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(8) + len(str(p[8]))
    sep_start, sep_stop, sep = p[7]
    node_sep = NodeTerm(sep_start, sep_stop, sep, NodeTerm.RDF_LITERAL)
    p[0] = Expression(lexstart, lexstop, [p[3], node_sep], str(p[1]))


def p_aggregate_5(p):
    '''
    aggregate : GROUP_CONCAT LPAREN DISTINCT expression SEMICOLON SEPARATOR EQ string RPAREN
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(9) + len(str(p[9]))
    sep_start, sep_stop, sep = p[8]
    node_sep = NodeTerm(sep_start, sep_stop, sep, NodeTerm.RDF_LITERAL)
    p[0] = Expression(lexstart, lexstop, [p[4], node_sep], str(p[1]), True)


# [128]
def p_iri_or_function_0(p):
    '''
    iri_or_function : iri
    '''
    p[0] = p[1]


def p_iri_or_function_1(p):
    '''
    iri_or_function : iri arg_list
    '''
    lexstart = p[1].lexstart
    _, lexstop, args, distinct = p[2]
    p[0] = Expression(lexstart, lexstop, [p[1]] + args, 'IRI_FUNC', distinct)


############################################################
# Group Graph Pattern
'''
[53] GroupGraphPattern ::= '{' ( SubSelect | GroupGraphPatternSub ) '}'
[54] GroupGraphPatternSub ::= TriplesBlock? ( GraphPatternNotTriples '.'? TriplesBlock? )*
[55] TriplesBlock ::= TriplesSameSubjectPath ( '.' TriplesBlock? )?
[56] GraphPatternNotTriples ::= GroupOrUnionGraphPattern | OptionalGraphPattern | MinusGraphPattern | GraphGraphPattern | ServiceGraphPattern | Filter | Bind | InlineData
[57] OptionalGraphPattern ::= 'OPTIONAL' GroupGraphPattern
[58] GraphGraphPattern ::= 'GRAPH' VarOrIri GroupGraphPattern
[59] ServiceGraphPattern ::= 'SERVICE' 'SILENT'? VarOrIri GroupGraphPattern
[60] Bind ::= 'BIND' '(' Expression 'AS' Var ')'
[61] InlineData ::= 'VALUES' DataBlock
[62] DataBlock ::= InlineDataOneVar | InlineDataFull
[63] InlineDataOneVar ::= Var '{' DataBlockValue* '}'
[64] InlineDataFull ::= ( NIL | '(' Var* ')' ) '{' ( '(' DataBlockValue* ')' | NIL )* '}'
[65] DataBlockValue ::= iri | RDFLiteral | NumericLiteral | BooleanLiteral | 'UNDEF'
[66] MinusGraphPattern ::= 'MINUS' GroupGraphPattern
[67] GroupOrUnionGraphPattern ::= GroupGraphPattern ( 'UNION' GroupGraphPattern )*
[68] Filter ::= 'FILTER' Constraint
[69] Constraint ::= BrackettedExpression | BuiltInCall | FunctionCall
[70] FunctionCall ::= iri ArgList
'''
############################################################

# [53]
def p_group_graph_pattern_0(p):
    '''
    group_graph_pattern : LBRACE sub_select RBRACE
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(3) + len(str(p[3]))
    p[0] = GraphPattern(lexstart, lexstop, [p[2]], GraphPattern.SUB_SELECT)


def p_group_graph_pattern_1(p):
    '''
    group_graph_pattern : LBRACE group_graph_pattern_sub RBRACE
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(3) + len(str(p[3]))
    p[0] = GraphPattern(lexstart, lexstop, p[2], GraphPattern.GROUP)


# [54]
def p_group_graph_pattern_sub_0(p):
    '''
    group_graph_pattern_sub : group_graph_pattern_sub_more
    '''
    p[0] = p[1]


def p_group_graph_pattern_sub_1(p):
    '''
    group_graph_pattern_sub : triples_block group_graph_pattern_sub_more
    '''
    tb_start = p[1][0].lexstart
    tb_stop = p[1][-1].lexstop
    tb = GraphPattern(tb_start, tb_stop, p[1], GraphPattern.TRIPLES_BLOCK)
    p[0] = [tb] + p[2]


def p_group_graph_pattern_sub_more_0(p):
    '''
    group_graph_pattern_sub_more : empty
    '''
    p[0] = []


def p_group_graph_pattern_sub_more_1(p):
    '''
    group_graph_pattern_sub_more : graph_pattern_not_triples group_graph_pattern_sub_more
    '''
    p[0] = [p[1]] + p[2]


def p_group_graph_pattern_sub_more_2(p):
    '''
    group_graph_pattern_sub_more : graph_pattern_not_triples DOT group_graph_pattern_sub_more
    '''
    p[0] = [p[1]] + p[3]


def p_group_graph_pattern_sub_more_3(p):
    '''
    group_graph_pattern_sub_more : graph_pattern_not_triples triples_block group_graph_pattern_sub_more
    '''
    tb_start = p[2][0].lexstart
    tb_stop = p[2][-1].lexstop
    tb = GraphPattern(tb_start, tb_stop, p[2], GraphPattern.TRIPLES_BLOCK)
    p[0] = [p[1], tb] + p[3]


def p_group_graph_pattern_sub_more_4(p):
    '''
    group_graph_pattern_sub_more : graph_pattern_not_triples DOT triples_block group_graph_pattern_sub_more
    '''
    tb_start = p[3][0].lexstart
    tb_stop = p[3][-1].lexstop
    tb = GraphPattern(tb_start, tb_stop, p[3], GraphPattern.TRIPLES_BLOCK)
    p[0] = [p[1], tb] + p[4]


# [55]
def p_triples_block_0(p):
    '''
    triples_block : triples_same_subject_path
                  | triples_same_subject_path DOT
    '''
    p[0] = [p[1]]


def p_triples_block_1(p):
    '''
    triples_block : triples_same_subject_path DOT triples_block
    '''
    p[0] = [p[1]] + p[3]


# [56]
def p_graph_pattern_not_triples(p):
    '''
    graph_pattern_not_triples : group_or_union_graph_pattern
                              | optional_graph_pattern
                              | minus_graph_pattern
                              | graph_graph_pattern
                              | service_graph_pattern
                              | filter
                              | bind
                              | inline_data
    '''
    p[0] = p[1]


# [57]
def p_optional_graph_pattern(p):
    '''
    optional_graph_pattern : OPTIONAL group_graph_pattern
    '''
    lexstart = p.lexpos(1)
    lexstop = p[2].lexstop
    p[0] = GraphPattern(lexstart, lexstop, [p[2]], GraphPattern.OPTIONAL)


# [58]
def p_graph_graph_pattern(p):
    '''
    graph_graph_pattern : GRAPH var_or_iri group_graph_pattern
    '''
    lexstart = p.lexpos(1)
    lexstop = p[3].lexstop
    p[0] = GraphPattern(lexstart, lexstop, [p[2], p[3]], GraphPattern.GRAPH)


# [59]
def p_service_graph_pattern_0(p):
    '''
    service_graph_pattern : SERVICE var_or_iri group_graph_pattern
    '''
    lexstart = p.lexpos(1)
    lexstop = p[3].lexstop
    p[0] = GraphPattern(lexstart, lexstop, [p[2], p[3]], GraphPattern.SERVICE)


def p_service_graph_pattern_1(p):
    '''
    service_graph_pattern : SERVICE SILENT var_or_iri group_graph_pattern
    '''
    lexstart = p.lexpos(1)
    lexstop = p[4].lexstop
    p[0] = GraphPattern(
        lexstart, lexstop, [p[3], p[4]], GraphPattern.SERVICE, True
    )


# [60]
def p_bind(p):
    '''
    bind : BIND LPAREN expression AS var RPAREN
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(6) + len(str(p[6]))
    p[0] = GraphPattern(lexstart, lexstop, [p[3], p[5]], GraphPattern.BIND)


# [61]
def p_inline_data(p):
    '''
    inline_data : VALUES data_block
    '''
    lexstart = p.lexpos(1)
    _, lexstop, data_block = p[2]
    p[0] = GraphPattern(
        lexstart, lexstop, data_block, GraphPattern.INLINE_DATA
    )


# [62]
def p_data_block(p):
    '''
    data_block : inline_data_one_var
               | inline_data_full
    '''
    p[0] = p[1]


# [63]
def p_inline_data_one_var(p):
    '''
    inline_data_one_var : var LBRACE data_block_value_more RBRACE
    '''
    lexstart = p[1].lexstart
    lexstop = p.lexpos(4) + len(str(p[4]))
    data_block = [[p[1]]] + [[x] for x in p[3]]
    p[0] = lexstart, lexstop, data_block


def p_data_block_value_more_0(p):
    '''
    data_block_value_more : empty
    '''
    p[0] = []


def p_data_block_value_more_1(p):
    '''
    data_block_value_more : data_block_value data_block_value_more
    '''
    p[0] = [p[1]] + p[2]


# [64]
def p_inline_data_full_0(p):
    '''
    inline_data_full : NIL LBRACE inline_data_full_more RBRACE
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(4) + len(str(p[4]))
    data_block = [[]] + p[3]
    p[0] = lexstart, lexstop, data_block


def p_inline_data_full_1(p):
    '''
    inline_data_full : LPAREN var_more RPAREN LBRACE inline_data_full_more RBRACE
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(6) + len(str(p[6]))
    data_block = [p[2]] + p[5]
    p[0] = lexstart, lexstop, data_block


def p_inline_data_full_more_0(p):
    '''
    inline_data_full_more : empty
    '''
    p[0] = []


def p_inline_data_full_more_1(p):
    '''
    inline_data_full_more : NIL inline_data_full_more
    '''
    p[0] = [[]] + p[2]


def p_inline_data_full_more_2(p):
    '''
    inline_data_full_more : LPAREN data_block_value_more RPAREN inline_data_full_more
    '''
    p[0] = [p[2]] + p[4]


def p_var_more_0(p):
    '''
    var_more : empty
    '''
    p[0] = []


def p_var_more_1(p):
    '''
    var_more : var var_more
    '''
    p[0] = [p[1]] + p[2]


# [65]
def p_data_block_value_0(p):
    '''
    data_block_value : iri
                     | rdf_literal
                     | numeric_literal
                     | boolean_literal
    '''
    p[0] = p[1]


def p_data_block_value_1(p):
    '''
    data_block_value : UNDEF
    '''
    p[0] = rule_node_term(p, NodeTerm.SPECIAL)


# [66]
def p_minus_graph_pattern(p):
    '''
    minus_graph_pattern : MINUS group_graph_pattern
    '''
    lexstart = p.lexpos(1)
    lexstop = p[2].lexstop
    p[0] = GraphPattern(lexstart, lexstop, [p[2]], GraphPattern.MINUS)


# [67]
def p_group_or_union_graph_pattern(p):
    '''
    group_or_union_graph_pattern : group_graph_pattern group_or_union_graph_pattern_more
    '''
    if not p[2]:
        p[0] = p[1]
        return

    operands = [p[1]] + p[2]
    lexstart = operands[0].lexstart
    lexstop = operands[-1].lexstop
    p[0] = GraphPattern(lexstart, lexstop, operands, GraphPattern.UNION)


def p_group_or_union_graph_pattern_more_0(p):
    '''
    group_or_union_graph_pattern_more : empty
    '''
    p[0] = []


def p_group_or_union_graph_pattern_more_1(p):
    '''
    group_or_union_graph_pattern_more : UNION group_graph_pattern group_or_union_graph_pattern_more
    '''
    p[0] = [p[2]] + p[3]


# [68]
def p_filter(p):
    '''
    filter : FILTER constraint
    '''
    lexstart = p.lexpos(1)
    lexstop = p[2].lexstop
    p[0] = GraphPattern(lexstart, lexstop, [p[2]], GraphPattern.FILTER)


# [69]
def p_constraint(p):
    '''
    constraint : bracketted_expression
               | built_in_call
               | function_call
    '''
    p[0] = p[1]


# [70]
def p_function_call(p):
    '''
    function_call : iri arg_list
    '''
    lexstart = p[1].lexstart
    _, lexstop, args, distinct = p[2]
    p[0] = Expression(lexstart, lexstop, [p[1]] + args, 'IRI_FUNC', distinct)


############################################################
# SubSelect
'''
[8] SubSelect ::= SelectClause WhereClause SolutionModifier ValuesClause
[9] SelectClause ::= 'SELECT' ( 'DISTINCT' | 'REDUCED' )? ( ( Var | ( '(' Expression 'AS' Var ')' ) )+ | '*' )
[17] WhereClause ::= 'WHERE'? GroupGraphPattern
[18] SolutionModifier ::= GroupClause? HavingClause? OrderClause? LimitOffsetClauses?
[19] GroupClause ::= 'GROUP' 'BY' GroupCondition+
[20] GroupCondition ::= BuiltInCall | FunctionCall | '(' Expression ( 'AS' Var )? ')' | Var
[21] HavingClause ::= 'HAVING' HavingCondition+
[22] HavingCondition ::= Constraint
[23] OrderClause ::= 'ORDER' 'BY' OrderCondition+
[24] OrderCondition ::= ( ( 'ASC' | 'DESC' ) BrackettedExpression ) | ( Constraint | Var )
[25] LimitOffsetClauses ::= LimitClause OffsetClause? | OffsetClause LimitClause?
[26] LimitClause ::= 'LIMIT' INTEGER
[27] OffsetClause ::= 'OFFSET' INTEGER
[28] ValuesClause ::= ( 'VALUES' DataBlock )?
'''
############################################################

# [8]
def p_sub_select(p):
    '''
    sub_select : select_clause where_clause solution_modifier values_clause
    '''
    lexstart, _, select_modifier, target = p[1]
    _, lexstop, pattern = p[2]
    modifier_dict = dict()
    values = None
    if p[3] is not None:
        _, lexstop, modifier_dict = p[3]        
    if p[4] is not None:
        _, lexstop, values = p[4]

    p[0] = Query(
        lexstart, lexstop, Query.SUB_SELECT,
        select_modifier=select_modifier,
        target=target,
        pattern=pattern,
        values=values,
        **modifier_dict,
    )


# [9]
def p_select_clause_0(p):
    '''
    select_clause : SELECT select_modifier STAR
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(3) + len(str(p[3]))
    proj = rule_node_term(p, NodeTerm.SPECIAL, 3)
    p[0] = lexstart, lexstop, p[2], proj


def p_select_clause_1(p):
    '''
    select_clause : SELECT select_modifier select_projection select_projection_more
    '''
    proj = [p[3]] + p[4]
    lexstart = p.lexpos(1)
    lexstop = proj[-1].lexstop
    p[0] = lexstart, lexstop, p[2], proj


def p_select_modifier_0(p):
    '''
    select_modifier : empty
    '''
    p[0] = None


def p_select_modifier_1(p):
    '''
    select_modifier : DISTINCT
                    | REDUCED
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(1) + len(str(p[1]))
    p[0] = ComponentWrapper(lexstart, lexstop, str(p[1]).upper())


def p_select_projection_0(p):
    '''
    select_projection : var
    '''
    p[0] = p[1]


def p_select_projection_1(p):
    '''
    select_projection : LPAREN expression AS var RPAREN
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(5) + len(str(p[5]))
    p[0] = ComponentWrapper(lexstart, lexstop, [p[2], p[4]])


def p_select_projection_more_0(p):
    '''
    select_projection_more : empty
    '''
    p[0] = []


def p_select_projection_more_1(p):
    '''
    select_projection_more : select_projection select_projection_more
    '''
    p[0] = [p[1]] + p[2]


# [17]
def p_where_clause_0(p):
    '''
    where_clause : group_graph_pattern
    '''
    p[0] = p[1].lexstart, p[1].lexstop, p[1]


def p_where_clause_1(p):
    '''
    where_clause : WHERE group_graph_pattern
    '''
    lexstart = p.lexpos(1)
    lexstop = p[2].lexstop
    p[0] = lexstart, lexstop, p[2]


# [18]
def p_solution_modifier(p):
    '''
    solution_modifier : group_clause_opt having_clause_opt order_clause_opt limit_offset_clauses_opt
    '''
    lo1, lo2, need_swap = p[4] if p[4] is not None else (None, None, False)
    tmp_list = p[1:4] + [lo1, lo2]
    if all(x is None for x in tmp_list):
        p[0] = None
        return

    lexstart = next(x.lexstart for x in tmp_list if x is not None)
    lexstop = next(x.lexstop for x in reversed(tmp_list) if x is not None)
    if need_swap:
        tmp_list[3], tmp_list[4] = tmp_list[4], tmp_list[3]
    modifier_dict = {
        'group_by': tmp_list[0],
        'having': tmp_list[1],
        'order_by': tmp_list[2],
        'limit': tmp_list[3],
        'offset': tmp_list[4],
    }
    p[0] = lexstart, lexstop, modifier_dict


def p_group_clause_opt_0(p):
    '''
    group_clause_opt : empty
    '''
    p[0] = None


def p_group_clause_opt_1(p):
    '''
    group_clause_opt : group_clause
    '''
    p[0] = p[1]


def p_having_clause_opt_0(p):
    '''
    having_clause_opt : empty
    '''
    p[0] = None


def p_having_clause_opt_1(p):
    '''
    having_clause_opt : having_clause
    '''
    p[0] = p[1]


def p_order_clause_opt_0(p):
    '''
    order_clause_opt : empty
    '''
    p[0] = None


def p_order_clause_opt_1(p):
    '''
    order_clause_opt : order_clause
    '''
    p[0] = p[1]


def p_limit_offset_clauses_opt_0(p):
    '''
    limit_offset_clauses_opt : empty
    '''
    p[0] = None


def p_limit_offset_clauses_opt_1(p):
    '''
    limit_offset_clauses_opt : limit_offset_clauses
    '''
    p[0] = p[1]


# [19]
def p_group_clause(p):
    '''
    group_clause : GROUP BY group_condition group_condition_more
    '''
    cond_list = [p[3]] + p[4]
    lexstart = p.lexpos(1)
    lexstop = cond_list[-1].lexstop
    p[0] = ComponentWrapper(lexstart, lexstop, cond_list)


def p_group_condition_more_0(p):
    '''
    group_condition_more : empty
    '''
    p[0] = []


def p_group_condition_more_1(p):
    '''
    group_condition_more : group_condition group_condition_more
    '''
    p[0] = [p[1]] + p[2]


# [20]
def p_group_condition_0(p):
    '''
    group_condition : built_in_call
                    | function_call
                    | var
    '''
    p[0] = p[1]


def p_group_condition_1(p):
    '''
    group_condition : LPAREN expression RPAREN
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(3) + len(str(p[3]))
    p[0] = Expression(lexstart, lexstop, [p[2]])


def p_group_condition_2(p):
    '''
    group_condition : LPAREN expression AS var RPAREN
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(5) + len(str(p[5]))
    p[0] = ComponentWrapper(lexstart, lexstop, [p[2], p[4]])


# [21]
def p_having_clause(p):
    '''
    having_clause : HAVING having_condition having_condition_more
    '''
    cond_list = [p[2]] + p[3]
    lexstart = p.lexpos(1)
    lexstop = cond_list[-1].lexstop
    p[0] = ComponentWrapper(lexstart, lexstop, cond_list)


def p_having_condition_more_0(p):
    '''
    having_condition_more : empty
    '''
    p[0] = []


def p_having_condition_more_1(p):
    '''
    having_condition_more : having_condition having_condition_more
    '''
    p[0] = [p[1]] + p[2]


# [22]
def p_having_condition(p):
    '''
    having_condition : constraint
    '''
    p[0] = p[1]


# [23]
def p_order_clause(p):
    '''
    order_clause : ORDER BY order_condition order_condition_more
    '''
    cond_list = [p[3]] + p[4]
    lexstart = p.lexpos(1)
    lexstop = cond_list[-1].lexstop
    p[0] = ComponentWrapper(lexstart, lexstop, cond_list)


def p_order_condition_more_0(p):
    '''
    order_condition_more : empty
    '''
    p[0] = []


def p_order_condition_more_1(p):
    '''
    order_condition_more : order_condition order_condition_more
    '''
    p[0] = [p[1]] + p[2]


# [24]
def p_order_condition_0(p):
    '''
    order_condition : ASC bracketted_expression
                    | DESC bracketted_expression
    '''
    lexstart = p.lexpos(1)
    lexstop = p[2].lexstop
    cond = [str(p[1]).upper(), p[2]]
    p[0] = ComponentWrapper(lexstart, lexstop, cond)


def p_order_condition_1(p):
    '''
    order_condition : constraint
                    | var
    '''
    p[0] = p[1]


# [25]
def p_limit_offset_clauses_0(p):
    '''
    limit_offset_clauses : limit_clause
    '''
    p[0] = p[1], None, False


def p_limit_offset_clauses_1(p):
    '''
    limit_offset_clauses : offset_clause
    '''
    p[0] = None, p[1], False



def p_limit_offset_clauses_2(p):
    '''
    limit_offset_clauses : limit_clause offset_clause
    '''
    p[0] = p[1], p[2], False


def p_limit_offset_clauses_3(p):
    '''
    limit_offset_clauses : offset_clause limit_clause
    '''
    p[0] = p[1], p[2], True


# [26]
def p_limit_clause(p):
    '''
    limit_clause : LIMIT INTEGER
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(2) + len(str(p[2]))
    node = rule_node_term(p, NodeTerm.INTEGER, 2)
    p[0] = ComponentWrapper(lexstart, lexstop, node)


# [27]
def p_offset_clause(p):
    '''
    offset_clause : OFFSET INTEGER
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(2) + len(str(p[2]))
    node = rule_node_term(p, NodeTerm.INTEGER, 2)
    p[0] = ComponentWrapper(lexstart, lexstop, node)


# [28]
def p_values_clause_0(p):
    '''
    values_clause : empty
    '''
    p[0] = None


def p_values_clause_1(p):
    '''
    values_clause : VALUES data_block
    '''
    lexstart = p.lexpos(1)
    _, lexstop, data_block = p[2]
    clause = ComponentWrapper(lexstart, lexstop, data_block)
    p[0] = lexstart, lexstop, clause


############################################################
# Query
'''
[1] QueryUnit ::= Query
[2] Query ::= Prologue
              ( SelectQuery | ConstructQuery | DescribeQuery | AskQuery )
              ValuesClause
[4] Prologue ::= ( BaseDecl | PrefixDecl )*
[5] BaseDecl ::= 'BASE' IRIREF
[6] PrefixDecl ::= 'PREFIX' PNAME_NS IRIREF

[7] SelectQuery ::= SelectClause DatasetClause* WhereClause SolutionModifier
[13] DatasetClause ::= 'FROM' ( DefaultGraphClause | NamedGraphClause )
[14] DefaultGraphClause ::= SourceSelector
[15] NamedGraphClause ::= 'NAMED' SourceSelector
[16] SourceSelector ::= iri

[11] DescribeQuery ::= 'DESCRIBE' ( VarOrIri+ | '*' ) DatasetClause* WhereClause? SolutionModifier
[12] AskQuery ::= 'ASK' DatasetClause* WhereClause SolutionModifier

[10] ConstructQuery ::= 'CONSTRUCT' ( ConstructTemplate DatasetClause* WhereClause SolutionModifier | DatasetClause* 'WHERE' '{' TriplesTemplate? '}' SolutionModifier )
[52] TriplesTemplate ::= TriplesSameSubject ( '.' TriplesTemplate? )?
[73] ConstructTemplate ::= '{' ConstructTriples? '}'
[74] ConstructTriples ::= TriplesSameSubject ( '.' ConstructTriples? )?
[75] TriplesSameSubject ::= VarOrTerm PropertyListNotEmpty | TriplesNode PropertyList
[76] PropertyList ::= PropertyListNotEmpty?
'''
############################################################

def rule_component_list(p, idx=1, idx_more=2):
    lexstart, lexstop, symbol = p[idx].lexstart, p[idx].lexstop, p[idx]
    if p[idx_more] is None:
        return lexstart, lexstop, [symbol]

    _, lexstop, symbol_more = p[idx_more]
    return lexstart, lexstop, [symbol] + symbol_more


# [1]
def p_query_unit(p):
    '''
    query_unit : query
    '''
    p[0] = p[1]


# [2]
def p_query(p):
    '''
    query : prologue select_query values_clause
          | prologue construct_query values_clause
          | prologue describe_query values_clause
          | prologue ask_query values_clause
    '''
    lexstart, lexstop, typ, kwargs = p[2]
    prologue = None
    values = None
    if p[1] is not None:
        lexstart, _, prologue = p[1]
    if p[3] is not None:
        _, lexstop, values = p[3]

    p[0] = Query(
        lexstart, lexstop, typ,
        prologue=prologue,
        values=values,
        raw_sparql=p.lexer.lexdata,
        **kwargs,
    )


# [4]
def p_prologue_0(p):
    '''
    prologue : empty
    '''
    p[0] = None


def p_prologue_1(p):
    '''
    prologue : base_decl prologue
    '''
    p[0] = rule_component_list(p)


def p_prologue_2(p):
    '''
    prologue : prefix_decl prologue
    '''
    p[0] = rule_component_list(p)


# [5]
def p_base_decl(p):
    '''
    base_decl : BASE IRIREF
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(2) + len(str(p[2]))
    iri = rule_node_term(p, NodeTerm.IRIREF, 2)
    decl = [str(p[1]).upper(), iri]
    p[0] = ComponentWrapper(lexstart, lexstop, decl)


# [6]
def p_prefix_decl(p):
    '''
    prefix_decl : PREFIX PNAME_NS IRIREF
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(3) + len(str(p[3]))
    pn = rule_node_term(p, NodeTerm.PREFIXED_NAME, 2)
    iri = rule_node_term(p, NodeTerm.IRIREF, 3)
    decl = [str(p[1]).upper(), pn, iri]
    p[0] = ComponentWrapper(lexstart, lexstop, decl)


# [7]
def p_select_query(p):
    '''
    select_query : select_clause dataset_clause_more where_clause solution_modifier
    '''
    lexstart, _, select_modifier, target = p[1]
    dataset = None
    _, lexstop, pattern = p[3]
    modifier_dict = dict()
    
    if p[2] is not None:
        _, _, dataset = p[2]
    if p[4] is not None:
        _, lexstop, modifier_dict = p[4]

    p[0] = lexstart, lexstop, Query.SELECT, {
        'select_modifier': select_modifier, 
        'target': target,
        'dataset': dataset,
        'pattern': pattern,
        **modifier_dict,
    }


def p_dataset_clause_more_0(p):
    '''
    dataset_clause_more : empty
    '''
    p[0] = None


def p_dataset_clause_more_1(p):
    '''
    dataset_clause_more : dataset_clause dataset_clause_more
    '''
    p[0] = rule_component_list(p)


# [13]
def p_dataset_clause_0(p):
    '''
    dataset_clause : FROM default_graph_clause
    '''
    lexstart = p.lexpos(1)
    lexstop = p[2].lexstop
    clause = [str(p[1]).upper(), p[2]]
    p[0] = ComponentWrapper(lexstart, lexstop, clause)


def p_dataset_clause_1(p):
    '''
    dataset_clause : FROM named_graph_clause
    '''
    lexstart = p.lexpos(1)
    _, lexstop, named, iri = p[2]
    clause = [str(p[1]).upper(), named, iri]
    p[0] = ComponentWrapper(lexstart, lexstop, clause)
    

# [14]
def p_default_graph_clause(p):
    '''
    default_graph_clause : source_selector
    '''
    p[0] = p[1]


# [15]
def p_named_graph_clause(p):
    '''
    named_graph_clause : NAMED source_selector
    '''
    lexstart = p.lexpos(1)
    lexstop = p[2].lexstop
    p[0] = lexstart, lexstop, str(p[1]).upper(), p[2]


# [16]
def p_source_selector(p):
    '''
    source_selector : iri
    '''
    p[0] = p[1]


# [11]
def p_describe_query(p):
    '''
    describe_query : DESCRIBE describe_target dataset_clause_more where_clause_opt solution_modifier
    '''
    lexstart = p.lexpos(1)
    _, lexstop, target = p[2]
    dataset = None
    pattern = None
    modifier_dict = dict()

    if p[3] is not None:
        _, lexstop, dataset = p[3]
    if p[4] is not None:
        _, lexstop, pattern = p[4]
    if p[5] is not None:
        _, lexstop, modifier_dict = p[5]  

    p[0] = lexstart, lexstop, Query.DESCRIBE, {
        'target': target,
        'dataset': dataset,
        'pattern': pattern,
        **modifier_dict,
    }


def p_describe_target_0(p):
    '''
    describe_target : STAR
    '''
    node = rule_node_term(p, NodeTerm.SPECIAL)
    p[0] = node.lexstart, node.lexstop, node


def p_describe_target_1(p):
    '''
    describe_target : var_or_iri var_or_iri_more
    '''
    target = [p[1]] + p[2]
    lexstart = target[0].lexstart
    lexstop = target[-1].lexstop
    p[0] = lexstart, lexstop, target


def p_var_or_iri_more_0(p):
    '''
    var_or_iri_more : empty
    '''
    p[0] = []


def p_var_or_iri_more_1(p):
    '''
    var_or_iri_more : var_or_iri var_or_iri_more
    '''
    p[0] = [p[1]] + p[2]


def p_where_clause_opt_0(p):
    '''
    where_clause_opt : empty
    '''
    p[0] = None


def p_where_clause_opt_1(p):
    '''
    where_clause_opt : where_clause
    '''
    p[0] = p[1]


# [12]
def p_ask_query(p):
    '''
    ask_query : ASK dataset_clause_more where_clause solution_modifier
    '''
    lexstart = p.lexpos(1)
    dataset = None
    _, lexstop, pattern = p[3]
    modifier_dict = dict()

    if p[2] is not None:
        _, _, dataset = p[2]
    if p[4] is not None:
        _, lexstop, modifier_dict = p[4]

    p[0] = lexstart, lexstop, Query.ASK, {
        'dataset': dataset,
        'pattern': pattern,
        **modifier_dict,
    }


# [10]
def p_construct_query_0(p):
    '''
    construct_query : CONSTRUCT construct_template dataset_clause_more where_clause solution_modifier
    '''
    lexstart = p.lexpos(1)
    target = p[2]
    dataset = None
    _, lexstop, pattern = p[4]
    modifier_dict = dict()

    if p[3] is not None:
        _, _, dataset = p[3]
    if p[5] is not None:
        _, lexstop, modifier_dict = p[5]

    p[0] = lexstart, lexstop, Query.CONSTRUCT, {
        'target': target,
        'dataset': dataset,
        'pattern': pattern,
        **modifier_dict,
    }


def p_construct_query_1(p):
    '''
    construct_query : CONSTRUCT dataset_clause_more WHERE LBRACE triples_template_opt RBRACE solution_modifier
    '''
    pat_start = p.lexpos(4)
    pat_stop = p.lexpos(6) + len(str(p[6]))
    inner_pat = [p[5]] if p[5] is not None else []
    pattern = GraphPattern(pat_start, pat_stop, inner_pat, GraphPattern.TRIPLES_BLOCK)

    lexstart = p.lexpos(1)
    dataset = None
    modifier_dict = dict()

    if p[2] is not None:
        _, _, dataset = p[2]
    if p[7] is not None:
        _, lexstop, modifier_dict = p[7]

    p[0] = lexstart, lexstop, Query.CONSTRUCT, {
        'dataset': dataset,
        'pattern': pattern,
        **modifier_dict,
    }


def p_triples_template_opt_0(p):
    '''
    triples_template_opt : empty
    '''
    p[0] = None


def p_triples_template_opt_1(p):
    '''
    triples_template_opt : triples_template
    '''
    lexstart = p[1][0].lexstart
    lexstop = p[1][-1].lexstop
    p[0] = GraphPattern(lexstart, lexstop, p[1], GraphPattern.TRIPLES_BLOCK)


# [52]
def p_triples_template_0(p):
    '''
    triples_template : triples_same_subject
                     | triples_same_subject DOT
    '''
    p[0] = [p[1]]


def p_triples_template_1(p):
    '''
    triples_template : triples_same_subject DOT triples_template
    '''
    p[0] = [p[1]] + p[3]


# [73]
def p_construct_template_0(p):
    '''
    construct_template : LBRACE RBRACE
    '''
    lexstart = p.lexpos(1)
    lexstop = p.lexpos(2) + len(str(p[2]))
    p[0] = GraphPattern(lexstart, lexstop, [], GraphPattern.GROUP)


def p_construct_template_1(p):
    '''
    construct_template : LBRACE construct_triples RBRACE
    '''
    tb_start = p[2][0].lexstart
    tb_stop = p[2][-1].lexstop
    tb = GraphPattern(tb_start, tb_stop, p[2], GraphPattern.TRIPLES_BLOCK)

    lexstart = p.lexpos(1)
    lexstop = p.lexpos(3) + len(str(p[3]))
    p[0] = GraphPattern(lexstart, lexstop, [tb], GraphPattern.GROUP)


# [74]
def p_construct_triples_0(p):
    '''
    construct_triples : triples_same_subject
                      | triples_same_subject DOT
    '''
    p[0] = [p[1]]


def p_construct_triples_1(p):
    '''
    construct_triples : triples_same_subject DOT construct_triples
    '''
    p[0] = [p[1]] + p[3]


# [75]
def p_triples_same_subject_0(p):
    '''
    triples_same_subject : var_or_term property_list_not_empty
                         | triples_node property_list
    '''
    lexstart = p[1].lexstart
    lexstop = (
        p[2][-1][1][-1].lexstop
        if p[2] is not None else p[1].lexstop
    )
    p[0] = TriplesPath(lexstart, lexstop, p[1], p[2])


# [76]
def p_property_list_0(p):
    '''
    property_list : empty
    '''
    p[0] = None


def p_property_list_1(p):
    '''
    property_list : property_list_not_empty
    '''
    p[0] = p[1]


############################################################
# Update
'''
[3] UpdateUnit ::= Update
[29] Update ::= Prologue ( Update1 ( ';' Update )? )?
[30] Update1 ::= Load | Clear | Drop | Add | Move | Copy | Create | InsertData | DeleteData | DeleteWhere | Modify
[31] Load ::= 'LOAD' 'SILENT'? iri ( 'INTO' GraphRef )?
[32] Clear ::= 'CLEAR' 'SILENT'? GraphRefAll
[33] Drop ::= 'DROP' 'SILENT'? GraphRefAll
[34] Create ::= 'CREATE' 'SILENT'? GraphRef
[35] Add ::= 'ADD' 'SILENT'? GraphOrDefault 'TO' GraphOrDefault
[36] Move ::= 'MOVE' 'SILENT'? GraphOrDefault 'TO' GraphOrDefault
[37] Copy ::= 'COPY' 'SILENT'? GraphOrDefault 'TO' GraphOrDefault
[38] InsertData ::= 'INSERT DATA' QuadData
[39] DeleteData ::= 'DELETE DATA' QuadData
[40] DeleteWhere ::= 'DELETE WHERE' QuadPattern
[41] Modify ::= ( 'WITH' iri )? ( DeleteClause InsertClause? | InsertClause ) UsingClause* 'WHERE' GroupGraphPattern
[42] DeleteClause ::= 'DELETE' QuadPattern
[43] InsertClause ::= 'INSERT' QuadPattern
[44] UsingClause ::= 'USING' ( iri | 'NAMED' iri )
[45] GraphOrDefault ::= 'DEFAULT' | 'GRAPH'? iri
[46] GraphRef ::= 'GRAPH' iri
[47] GraphRefAll ::= GraphRef | 'DEFAULT' | 'NAMED' | 'ALL'
[48] QuadPattern ::= '{' Quads '}'
[49] QuadData ::= '{' Quads '}'
[50] Quads ::= TriplesTemplate? ( QuadsNotTriples '.'? TriplesTemplate? )*
[51] QuadsNotTriples ::= 'GRAPH' VarOrIri '{' TriplesTemplate? '}'
'''
############################################################

pass


def parse(sparql, debug=False, start='query_unit'):
    lexer = lex.lex(module=sparql_lex, debug=debug)
    parser = yacc.yacc(start=start, debug=debug)
    QueryComponent.DEBUG = debug
    return parser.parse(sparql, lexer=lexer)


if __name__ == '__main__':   
    data = 'PREFIX  rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\nPREFIX  wde:  <http://www.wikidata.org/entity/>\n\nSELECT  *\nWHERE\n  { _:b0  rdf:rest   ( wde:Q56061 ) }'
    start = 'query_unit'

    # print(parse(data, debug=False))
    print(parse(data, debug=True))

