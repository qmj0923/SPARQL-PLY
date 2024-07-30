# SPARQL-PLY

A SPARQL 1.1 query parser implemented with the [PLY](https://github.com/dabeaz/ply) library.

# Data Structure

This parser defines several data structures, referred to as query components, to represent the syntax of SPARQL queries. Here are some figures to illustrate them. See the source code in `sparql_ply/components.py` for more details.

### Query

![Query](doc/figure/Query.png)

### GraphPattern

![GraphPattern](doc/figure/GraphPattern.png)

### NodeTerm

![NodeTerm](doc/figure/NodeTerm.png)

### PropertyPath

![PropertyPath](doc/figure/PropertyPath.png)

### CollectionPath & BlankNodePath & TriplesPath

![TriplesPath](doc/figure/TriplesPath.png)

### Expression

![Expression](doc/figure/Expression.png)

# Usage

## Parse SPARQL Query

Parse a SPARQL query string into a query component.

```pycon
>>> from sparql_ply import parse_sparql
>>> query = parse_sparql('SELECT * WHERE { ?s ?p ?o }')
```

## Collect Query Component

Collect components of specific types in the query.

For example, the following code collects all the variables in the query.

```pycon
>>> from sparql_ply import parse_sparql
>>> from sparql_ply.components import NodeTerm
>>> from sparql_ply.util import collect_component
>>> query = parse_sparql('SELECT * WHERE { ?s ?p ?o }')
>>> collect_component(query, NodeTerm.VAR)
[?s, ?p, ?o]
```

## Expand Syntax Form

Expand syntax forms in SPARQL query according to [SPARQL 1.1 document](https://www.w3.org/TR/2013/REC-sparql11-query-20130321/#sparqlExpandForms).

The following syntax forms can be expanded:
1. [Keyword `a`](https://www.w3.org/TR/2013/REC-sparql11-query-20130321/#abbrevRdfType)
2. [Prefixed Names](https://www.w3.org/TR/2013/REC-sparql11-query-20130321/#prefNames) and [Relative IRIs](https://www.w3.org/TR/2013/REC-sparql11-query-20130321/#relIRIs)
3. [RDF Collections](https://www.w3.org/TR/2013/REC-sparql11-query-20130321/#collections)
4. [Blank Nodes](https://www.w3.org/TR/2013/REC-sparql11-query-20130321/#QSynBlankNodes)
5. [Predicate-Object Lists](https://www.w3.org/TR/2013/REC-sparql11-query-20130321/#predObjLists) and [Object Lists](https://www.w3.org/TR/2013/REC-sparql11-query-20130321/#objLists)


```pycon
>>> from sparql_ply.util import expand_syntax_form
>>> prefix_part = '\nPREFIX : <http://example.org/>\n'
>>> select_part = 'SELECT * WHERE {\n' + ':Person ?p [a ?q]' + '\n}'
>>> sparql = prefix_part + select_part
>>> print(expand_syntax_form(sparql))

PREFIX : <http://example.org/>
SELECT * WHERE {
<http://example.org/Person> ?p _:b0.
_:b0 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?q
}
```

