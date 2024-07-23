'''
Copyright (c) 2024 qmj0923
https://github.com/qmj0923/SPARQL-PLY
'''

from ply.lex import lex, TOKEN


SPARQL_KEYWORDS_CASE_SENSITIVE = {
    'a',
}

SPARQL_KEYWORDS_CASE_INSENSITIVE = {
    # query basic
    'BASE', 'PREFIX', 'SELECT', 'CONSTRUCT', 'DESCRIBE', 'ASK', 'DISTINCT', 'REDUCED', 'AS', 'WHERE', 'FROM', 'NAMED', 'GROUP', 'BY', 'HAVING', 'ORDER', 'ASC', 'DESC', 'LIMIT', 'OFFSET', 'VALUES', 'UNDEF',
    # group graph pattern
    'UNION', 'OPTIONAL', 'MINUS', 'GRAPH', 'SERVICE', 'FILTER', 'BIND',
    # literal and operator
    'TRUE', 'FALSE', 'NOT', 'IN', 'AND', 'OR',        
    # built-in call
    'COUNT', 'SUM', 'MIN', 'MAX', 'AVG', 'SAMPLE', 'GROUP_CONCAT', 'SEPARATOR', 'STR', 'LANG', 'LANGMATCHES', 'DATATYPE', 'BOUND', 'IRI', 'URI', 'BNODE', 'RAND', 'ABS', 'CEIL', 'FLOOR', 'ROUND', 'CONCAT', 'SUBSTR', 'STRLEN', 'REPLACE', 'UCASE', 'LCASE', 'ENCODE_FOR_URI', 'CONTAINS', 'STRSTARTS', 'STRENDS', 'STRBEFORE', 'STRAFTER', 'YEAR', 'MONTH', 'DAY', 'HOURS', 'MINUTES', 'SECONDS', 'TIMEZONE', 'TZ', 'NOW', 'UUID', 'STRUUID', 'MD5', 'SHA1', 'SHA256', 'SHA384', 'SHA512', 'COALESCE', 'IF', 'STRLANG', 'STRDT', 'SAMETERM', 'ISIRI', 'ISURI', 'ISBLANK', 'ISLITERAL', 'ISNUMERIC', 'REGEX', 'EXISTS',
    # update
    'LOAD', 'CLEAR', 'DROP', 'ADD', 'MOVE', 'COPY', 'CREATE', 'INSERT', 'DELETE', 'WITH', 'SILENT', 'DEFAULT', 'ALL', 'INTO', 'TO', 'DATA', 'USING',
}

tokens = [
    # -------------------------------------------------
    'LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET', 'LBRACE', 'RBRACE', 'SEMICOLON', 'COMMA', 'DOT', 'EQ', 'NE', 'GT', 'LT', 'LE', 'GE', 'BANG', 'TILDE', 'COLON', 'SC_OR', 'SC_AND', 'SC_PLUS', 'SC_MINUS', 'STAR', 'SLASH', 'PERCENT', 'DTYPE', 'AT', 'VBAR', 'CARAT', 'QMARK',
    # -------------------------------------------------
    'IRIREF', 'PNAME_NS', 'PNAME_LN', 'BLANK_NODE_LABEL', 'VAR1', 'VAR2', 'LANGTAG', 'INTEGER', 'DECIMAL', 'DOUBLE', 'INTEGER_POSITIVE', 'DECIMAL_POSITIVE', 'DOUBLE_POSITIVE', 'INTEGER_NEGATIVE', 'DECIMAL_NEGATIVE', 'DOUBLE_NEGATIVE', 'STRING_LITERAL1', 'STRING_LITERAL2', 'STRING_LITERAL_LONG1', 'STRING_LITERAL_LONG2', 'NIL', 'ANON',
    # -------------------------------------------------
    'KW_A',
] + list(SPARQL_KEYWORDS_CASE_INSENSITIVE)


# https://stackoverflow.com/questions/34712838/which-special-characters-must-be-escaped-when-using-python-regex-module-re
# https://stackoverflow.com/questions/399078/what-special-characters-must-be-escaped-in-regular-expressions

############################################################
#
#  Terminals (tokens)
#
############################################################
t_LPAREN = r'[(]'
t_RPAREN = r'[)]'
t_LBRACKET = r'[\[]'
t_RBRACKET = r'[\]]'
t_LBRACE = r'[{]'
t_RBRACE = r'[}]'

t_SEMICOLON = r'[;]'
t_COMMA = r'[,]'
t_DOT = r'[.]'
t_EQ = r'[=]'
t_NE = r'[!][=]'
t_GT = r'[>]'
t_LT = r'[<]'
t_LE = r'[<][=]'
t_GE = r'[>][=]'
t_BANG = r'[!]'
t_TILDE = r'[~]'
t_COLON = r'[:]'
t_SC_OR = r'[|][|]'  # special character
t_SC_AND = r'[&][&]'
t_SC_PLUS = r'[+]'
t_SC_MINUS = r'[\-]'
t_STAR = r'[*]'
t_SLASH = r'[/]'
t_PERCENT = r'[%]'
t_DTYPE = r'[\^][\^]'
t_AT = r'[@]'
t_VBAR = r'[|]'
t_CARAT = r'[\^]'
t_QMARK = r'[?]'

t_ignore = ' \t\r\f'
t_ignore_COMMENT = r'\#.*'


############################################################
#
#  Terminals (internal patterns)
#
############################################################
PAT_EXPONENT = r'[eE][+\-]?[0-9]+'
'''
[155] EXPONENT ::= [eE] [+-]? [0-9]+
'''
PAT_ECHAR = r'[\\][tbnrf\\\"\']'
'''
[160] ECHAR ::= '\' [tbnrf\"']
'''
PAT_WS = r'[\x20\x09\x0D\x0A]'
'''
[162] WS ::= #x20 | #x9 | #xD | #xA
WS ::= ' ' | '\t' | '\r' | '\n'
'''
INNER_PN_CHARS_BASE = (
    r'A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D'
    r'\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF'
    r'\uF900-\uFDCF\uFDF0-\uFFFD\U00010000-\U000EFFFF'
)
'''
[164] PN_CHARS_BASE ::= [A-Z] | [a-z] | [#x00C0-#x00D6] | [#x00D8-#x00F6]
    | [#x00F8-#x02FF] | [#x0370-#x037D] | [#x037F-#x1FFF] | [#x200C-#x200D]
    | [#x2070-#x218F] | [#x2C00-#x2FEF] | [#x3001-#xD7FF] | [#xF900-#xFDCF]
    | [#xFDF0-#xFFFD] | [#x10000-#xEFFFF]
'''
INNER_PN_CHARS_U = INNER_PN_CHARS_BASE + r'_'
'''
[165] PN_CHARS_U ::= PN_CHARS_BASE | '_'
'''
PAT_VARNAME = (
    r'([' + INNER_PN_CHARS_U + r'0-9])'
    r'([' + INNER_PN_CHARS_U + r'0-9\u00B7\u0300-\u036F\u203F-\u2040])*'
)
'''
[166] VARNAME ::= ( PN_CHARS_U | [0-9] )
    ( PN_CHARS_U | [0-9] | #x00B7 | [#x0300-#x036F] | [#x203F-#x2040] )*
'''
INNER_PN_CHARS = INNER_PN_CHARS_U + r'\-0-9\u00B7\u0300-\u036F\u203F-\u2040'
'''
[167] PN_CHARS ::= PN_CHARS_U | '-' | [0-9]
    | #x00B7 | [#x0300-#x036F] | [#x203F-#x2040]
'''
PAT_PN_PREFIX = (
    r'([' + INNER_PN_CHARS_BASE + r'])'
    r'(([' + INNER_PN_CHARS + r'.])*[' + INNER_PN_CHARS + r'])?'
)
'''
[168] PN_PREFIX ::= PN_CHARS_BASE ((PN_CHARS|'.')* PN_CHARS)?
'''
PAT_HEX = r'[0-9A-Fa-f]'
'''
[172] HEX ::= [0-9] | [A-F] | [a-f]
'''
PAT_PERCENT = r'%' + PAT_HEX + PAT_HEX
'''
[171] PERCENT ::= '%' HEX HEX
'''
PAT_PN_LOCAL_ESC = r'[\\][_~.\-!$&\'()*+,;=/?#@%]'
'''
[173] PN_LOCAL_ESC ::= '\' ( '_' | '~' | '.' | '-' | '!' | '$' | '&' | "'"
    | '(' | ')' | '*' | '+' | ',' | ';' | '=' | '/' | '?' | '#' | '@' | '%' )
'''
PAT_PLX = r'(' + PAT_PERCENT + r')|(' + PAT_PN_LOCAL_ESC + r')'
'''
[170] PLX ::= PERCENT | PN_LOCAL_ESC
'''
PAT_PN_LOCAL = (
    r'(([' + INNER_PN_CHARS_U + r':0-9])|' + PAT_PLX + r')'
    r'(' + (
        r'(([' + INNER_PN_CHARS + r'.:])|' + PAT_PLX + r')*'
        r'(([' + INNER_PN_CHARS + r':])|' + PAT_PLX + r')'
    ) + r')?'
)
'''
[169] PN_LOCAL ::= (PN_CHARS_U | ':' | [0-9] | PLX )
    ((PN_CHARS | '.' | ':' | PLX)* (PN_CHARS | ':' | PLX) )?
'''

############################################################
#
#  Terminals (for lexer)
#
############################################################
PAT_IRIREF = r'<([^<>\"{}|\^`\\\x00-\x20])*>'
'''
[139] IRIREF ::= '<' ([^<>"{}|^`\]-[#x00-#x20])* '>'
'''
PAT_PNAME_NS = r'(' + PAT_PN_PREFIX + r')?[:]'
'''
[140] PNAME_NS ::= PN_PREFIX? ':'
'''
PAT_PNAME_LN = r'(' + PAT_PNAME_NS + r')(' + PAT_PN_LOCAL + r')'
'''
[141] PNAME_LN ::= PNAME_NS PN_LOCAL
'''
PAT_BLANK_NODE_LABEL = (
    r'[_][:]([' + INNER_PN_CHARS_U + r'0-9])'
    r'(([' + INNER_PN_CHARS + r'.])*[' + INNER_PN_CHARS + r'])?'
)
'''
[142] BLANK_NODE_LABEL ::= '_:' ( PN_CHARS_U | [0-9] )
    ((PN_CHARS|'.')* PN_CHARS)?
'''
PAT_VAR1 = r'[?]' + PAT_VARNAME
'''
[143] VAR1 ::= '?' VARNAME
'''
PAT_VAR2 = r'[$]' + PAT_VARNAME
'''
[144] VAR2 ::= '$' VARNAME
'''
PAT_LANGTAG = r'[@][a-zA-Z]+([\-][a-zA-Z0-9]+)*'
'''
[145] LANGTAG ::= '@' [a-zA-Z]+ ('-' [a-zA-Z0-9]+)*
'''
PAT_INTEGER = r'[0-9]+'
'''
[146] INTEGER ::= [0-9]+
'''
PAT_DECIMAL = r'[0-9]*[.][0-9]+'
'''
[147] DECIMAL ::= [0-9]* '.' [0-9]+
'''
PAT_DOUBLE = (
    r'([0-9]+[.][0-9]*' + PAT_EXPONENT + r')'
    r'|([.][0-9]+' + PAT_EXPONENT + r')'
    r'|([0-9]+' + PAT_EXPONENT + r')'
)
'''
[148] DOUBLE ::= [0-9]+ '.' [0-9]* EXPONENT | '.' ([0-9])+ EXPONENT | ([0-9])+ EXPONENT
'''
PAT_INTEGER_POSITIVE = t_SC_PLUS + PAT_INTEGER
'''
[149] INTEGER_POSITIVE ::= '+' INTEGER
'''
PAT_DECIMAL_POSITIVE = t_SC_PLUS + PAT_DECIMAL
'''
[150] DECIMAL_POSITIVE ::= '+' DECIMAL
'''
PAT_DOUBLE_POSITIVE = t_SC_PLUS + PAT_DOUBLE
'''
[151] DOUBLE_POSITIVE ::= '+' DOUBLE
'''
PAT_INTEGER_NEGATIVE = t_SC_MINUS + PAT_INTEGER
'''
[152] INTEGER_NEGATIVE ::= '-' INTEGER
'''
PAT_DECIMAL_NEGATIVE = t_SC_MINUS + PAT_DECIMAL
'''
[153] DECIMAL_NEGATIVE ::= '-' DECIMAL
'''
PAT_DOUBLE_NEGATIVE = t_SC_MINUS + PAT_DOUBLE
'''
[154] DOUBLE_NEGATIVE ::= '-' DOUBLE
'''
PAT_STRING_LITERAL1 = r'\'(([^\x27\x5C\x0A\x0D])|(' + PAT_ECHAR + r'))*\''
'''
[156] STRING_LITERAL1 ::= "'" ( ([^#x27#x5C#xA#xD]) | ECHAR )* "'"
'''
PAT_STRING_LITERAL2 = r'\"(([^\x22\x5C\x0A\x0D])|(' + PAT_ECHAR + r'))*\"'
'''
[157] STRING_LITERAL2 ::= '"' ( ([^#x22#x5C#xA#xD]) | ECHAR )* '"'
'''
PAT_STRING_LITERAL_LONG1 = r'\'\'\'((\'|\'\')?([^\'\\]|' + PAT_ECHAR + r'))*\'\'\''
"""
[158] STRING_LITERAL_LONG1 ::= "'''" ( ( "'" | "''" )? ( [^'\] | ECHAR ) )* "'''"
"""
PAT_STRING_LITERAL_LONG2 = r'\"\"\"((\"|\"\")?([^\"\\]|' + PAT_ECHAR + r'))*\"\"\"'
'''
[159] STRING_LITERAL_LONG2 ::= '"""' ( ( '"' | '""' )? ( [^"\] | ECHAR ) )* '"""'
'''
PAT_NIL = t_LPAREN + r'(' + PAT_WS + r')*' + t_RPAREN
'''
[161] NIL ::= '(' WS* ')'
'''
PAT_ANON = t_LBRACKET + r'(' + PAT_WS + r')*' + t_RBRACKET
'''
[163] ANON ::= '[' WS* ']'
'''

############################################################
'''
The documetation of PLY says:
> When building the master regular expression, rules are added in the following order:
>   1. All tokens defined by functions are added in the same order as they appear in the lexer file.
>   2. Tokens defined by strings are added next by sorting them in order of decreasing regular expression length (longer expressions are added first).

In this case, if we define `t_PNAME_LN` by string instead of function, the
lexer will not recognize `wdt:P40` as `t_PNAME_LN` but recognize its prefix
`wdt` as `t_ID`.

Note that the order of the following tokens matters.
1. `t_PNAME_LN` <- `t_PNAME_NS`
2. `t_DOUBLE_XXX` <- `t_DECIMAL_XXX` <- `t_INTEGER_XXX`
3. `t_STRING_LITERAL_LONG` <- `t_STRING_LITERAL`
'''
############################################################

@TOKEN(PAT_IRIREF)
def t_IRIREF(t):
    return t


@TOKEN(PAT_PNAME_LN)
def t_PNAME_LN(t):
    return t


@TOKEN(PAT_PNAME_NS)
def t_PNAME_NS(t):
    return t


@TOKEN(PAT_BLANK_NODE_LABEL)
def t_BLANK_NODE_LABEL(t):
    return t


@TOKEN(PAT_VAR1)
def t_VAR1(t):
    return t


@TOKEN(PAT_VAR2)
def t_VAR2(t):
    return t


@TOKEN(PAT_LANGTAG)
def t_LANGTAG(t):
    return t


@TOKEN(PAT_DOUBLE)
def t_DOUBLE(t):
    return t


@TOKEN(PAT_DECIMAL)
def t_DECIMAL(t):
    return t

@TOKEN(PAT_INTEGER)
def t_INTEGER(t):
    return t


@TOKEN(PAT_DOUBLE_POSITIVE)
def t_DOUBLE_POSITIVE(t):
    return t


@TOKEN(PAT_DECIMAL_POSITIVE)
def t_DECIMAL_POSITIVE(t):
    return t


@TOKEN(PAT_INTEGER_POSITIVE)
def t_INTEGER_POSITIVE(t):
    return t


@TOKEN(PAT_DOUBLE_NEGATIVE)
def t_DOUBLE_NEGATIVE(t):
    return t


@TOKEN(PAT_DECIMAL_NEGATIVE)
def t_DECIMAL_NEGATIVE(t):
    return t


@TOKEN(PAT_INTEGER_NEGATIVE)
def t_INTEGER_NEGATIVE(t):
    return t


@TOKEN(PAT_STRING_LITERAL_LONG1)
def t_STRING_LITERAL_LONG1(t):
    return t


@TOKEN(PAT_STRING_LITERAL_LONG2)
def t_STRING_LITERAL_LONG2(t):
    return t


@TOKEN(PAT_STRING_LITERAL1)
def t_STRING_LITERAL1(t):
    return t


@TOKEN(PAT_STRING_LITERAL2)
def t_STRING_LITERAL2(t):
    return t


@TOKEN(PAT_NIL)
def t_NIL(t):
    return t


@TOKEN(PAT_ANON)
def t_ANON(t):
    return t


def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    if t.value.upper() in SPARQL_KEYWORDS_CASE_INSENSITIVE:
        t.type = t.value.upper()
    elif t.value == 'a':
        t.type = 'KW_A'
    else:
        raise NotImplementedError(
            f'[Lex] Unknown keyword {t.value} in line {t.lexer.lineno}'
            f' at position {t.lexpos}.'
        )
    return t


def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)


def t_error(t):
    if t is not None:
        raise TypeError(
            f'[Lex] Unknown text {t.value} in line {t.lexer.lineno}'
            f' at position {t.lexpos} in the following text: \n'
            f'{t.lexer.lexdata}'
        )
    else:
        raise TypeError(f'Unknown text')


if __name__ == '__main__':
    data = "PREFIX  rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\nPREFIX  wde:  <http://www.wikidata.org/entity/>\n\nSELECT  *\nWHERE\n  { _:b0  rdf:rest   ( wde:Q56061 ) }"
    lexer = lex()
    # lexer = lex(debug=True)

    lexer.input(data)
    while True:
        tok = lexer.token()
        if not tok:
            break
        print(tok)
    
