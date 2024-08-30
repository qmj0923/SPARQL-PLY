'''
Copyright (c) 2024 qmj0923
https://github.com/qmj0923/SPARQL-PLY

Replace substrings in a given string using nested replacement strategies.
'''

from __future__ import annotations
import itertools
from collections import defaultdict
from copy import deepcopy
from typing import (
    List, Tuple, Dict, Callable, Optional, Union, Any,
)


class ReplacerNode:
    '''
    A node in the replacer tree.
    '''

    def __init__(
        self, key: Optional[str], remainder: List[str],
        children: List[ReplacerNode]
    ):
        if len(remainder) != len(children) + 1:
            raise ValueError(
                'Remainder must have one more element than children.'
            )
        self.key = key
        self.remainder = remainder
        self.children = children

    def __repr__(self) -> str:
        return (
            f'ReplacerNode(key={self.key}, remainder={self.remainder}, '
            f'children={self.children})'
        )

    def replace(
        self, handler_dict: Dict[
            str, Union[
                str, ReplacerNode, 
                Callable[[ReplacerNode], ReplacerNode]
            ]
        ]
    ) -> ReplacerNode:
        '''
        Replace nodes in the replacer tree using the given handler dictionary.

        This method replaces nodes in the replacer tree based on the provided
        handler dictionary, which maps each key to a handler. A handler can be
        one of the following:
        - A string: For each node whose `key` matches the dictionary key, a
            new leaf node with this string will replace the node.
        - A ReplacerNode: For each node whose `key` matches the dictionary
            key, a copy of this `ReplacerNode` will replace the node.
        - A callable: For each node whose `key` matches the dictionary key,
            this function will be called with the node as an argument, and it
            should return a `ReplacerNode` that will replace the node.

        Parameters
        ----------
        handler_dict
            A dictionary mapping keys to handlers.

        Returns
        -------
        ReplacerNode
            The replaced replacer tree.

        Raises
        ------
        ValueError
            If an invalid handler type is encountered.
        '''
        def handle_node(node: ReplacerNode) -> ReplacerNode:
            return ReplacerNode(
                node.key, node.remainder.copy(),
                [
                    dfs(child, handler_dict)
                    for child in node.children
                ],
            )

        def dfs(node: ReplacerNode, handler_dict: Dict) -> ReplacerNode:
            if node.key not in handler_dict:
                return handle_node(node)

            target = handler_dict[node.key]
            if isinstance(target, str):
                return ReplacerFactory.create_leaf(target)
            elif isinstance(target, ReplacerNode):
                return deepcopy(target)
            elif callable(target):
                return target(handle_node(node))
            else:
                raise ValueError(
                    f'Invalid handler type: {type(target)}'
                )

        return dfs(self, handler_dict).compact()

    def compact(self) -> ReplacerNode:
        '''
        Compact the replacer tree by merging any node whose `key` is None with
        its parent node.
        '''
        def handle_node(node: ReplacerNode) -> ReplacerNode:
            new_remainder = [node.remainder[0]]
            new_children = []
            for i, child in enumerate(node.children):
                if child.key is None:
                    last = new_remainder.pop()
                    r = child.remainder.copy()
                    r[0] = last + r[0]
                    r[-1] = r[-1] + node.remainder[i + 1]
                    new_remainder.extend(r)
                    new_children.extend(child.children)
                else:
                    new_remainder.append(node.remainder[i + 1])
                    new_children.append(child)
            return ReplacerNode(node.key, new_remainder, new_children)

        def dfs(node: ReplacerNode) -> ReplacerNode:
            result = ReplacerNode(
                node.key, node.remainder.copy(),
                [dfs(child) for child in node.children],
            )
            return handle_node(result)

        return dfs(self)

    def get_key2spans(self) -> Dict[str, List[Tuple[int, int]]]:
        '''
        Return a dictionary mapping each key to a list of spans, where each
        span is a tuple of (start, stop) indices in the original string.
        '''
        def dfs(node: ReplacerNode):
            node_len = 0
            key2spans = defaultdict(list)
            for x, y in zip(node.remainder, node.children):
                node_len += len(x)
                key2spans_y, node_len_y = dfs(y)
                for key, span_list in key2spans_y.items():
                    key2spans[key].extend([
                        (start + node_len, stop + node_len)
                        for start, stop in span_list
                    ])
                node_len += node_len_y
            node_len += len(node.remainder[-1])

            if node.key is not None:
                key2spans[node.key].append((0, node_len))
            return key2spans, node_len

        key2spans, _ = dfs(self)
        result = {
            key: sorted(span_list, key=lambda x: (x[1], -x[0]))
            for key, span_list in key2spans.items()
        }
        return result

    def to_str(self) -> str:
        '''
        Return a string that the replacer tree represents.
        '''
        if not self.children:
            return self.remainder[0]
        else:
            result = ''
            for x, y in zip(self.remainder, self.children):
                result += x + y.to_str()
            result += self.remainder[-1]
            return result

    def to_indented_repr(self, indent: int = 2) -> str:
        '''
        Return a string representation of the replacer tree with indentation.
        '''
        def dfs(node: ReplacerNode, depth: int) -> str:
            result = (
                f'{" " * (indent * depth)}ReplaceNode(\n'
                f'{" " * (indent * (depth + 1))}key={node.key},\n'
                f'{" " * (indent * (depth + 1))}remainder={node.remainder},\n'
            )
            if not node.children:
                result += (
                    f'{" " * (indent * (depth + 1))}children=[],\n'
                    f'{" " * (indent * depth)})\n'
                )
                return result
            else:
                result += f'{" " * (indent * (depth + 1))}children=[\n'
                for child in node.children:
                    result += dfs(child, depth + 2)
                result += (
                    f'{" " * (indent * (depth + 1))}],\n'
                    f'{" " * (indent * depth)})\n'
                )
                return result

        return dfs(self, 0)


class ReplacerFactory:
    '''
    A factory class for creating replacer trees.
    '''

    @staticmethod
    def create_leaf(
        content: str, key: Optional[str] = ...
    ) -> ReplacerNode:
        '''
        Make a replacer leaf from a string and an optional key.
        '''
        if key is ...:
            key = content
        return ReplacerNode(key, [content], [])

    @staticmethod
    def create_tree(
        content: str, key2spans: Dict[str, List[Tuple[int, int]]],
        rank_key: Callable[[Optional[str]], Any] = lambda x: x,
    ) -> ReplacerNode:
        '''
        Make a replacer tree from a string and a dictionary of key to spans.
        
        For nodes with the same span, parent nodes always have smaller
        `rank_key(key)` values than their children.
        '''
        def check_two_span(span1, span2):
            '''
            Check if two spans are disjoint or one contains the other.
            '''
            start1, stop1 = span1
            start2, stop2 = span2
            return (
                start1 <= stop1 and start2 <= stop2
            ) and (
                stop1 <= start2 or stop2 <= start1
                or (start1 <= start2 and stop2 <= stop1)
                or (start2 <= start1 and stop1 <= stop2)
            )

        if any((
            not check_two_span(span1, span2)
            for span1, span2 in itertools.combinations(
                itertools.chain.from_iterable(key2spans.values()), 2
            )
        )):
            raise ValueError(
                'Each pair of spans must be disjoint or one contains the other.'
            )

        node_list = sorted(
            [
                (key, start, stop)
                for key, span_list in key2spans.items()
                for start, stop in span_list
            ],
            key=lambda x: (-x[2], x[1], rank_key(x[0])),
            reverse=True,
        )
        mono_stack = list()
        for key, start, stop in node_list:
            if (not mono_stack) or start >= mono_stack[-1][2]:
                node = ReplacerNode(key, [content[start:stop]], [])
            else:
                remainder_r = list()
                children_r = list()
                last_s0 = stop
                while mono_stack and start <= mono_stack[-1][1]:
                    cn, s0, s1 = mono_stack.pop()
                    remainder_r.append(content[s1:last_s0])
                    children_r.append(cn)
                    last_s0 = s0
                remainder_r.append(content[start:last_s0])
                node = ReplacerNode(
                    key, list(reversed(remainder_r)), list(reversed(children_r))
                )
            mono_stack.append((node, start, stop))

        remainder = list()
        children = list()
        last_s1 = 0
        for node, s0, s1 in mono_stack:
            remainder.append(content[last_s1:s0])
            children.append(node)
            last_s1 = s1
        remainder.append(content[last_s1:])
        root = ReplacerNode(None, remainder, children)
        return root


if __name__ == '__main__':
    '''
    Suppose we have a pseduo function call `foo(a, b)`.
    We can construct a replacer tree for it as follows.
    '''
    content = 'foo(a, b)'
    key2spans = {
        'foo': [(0, 9)],
        'a': [(4, 5)],
        'b': [(7, 8)],
    }
    rt1 = ReplacerFactory.create_tree(content, key2spans)

    '''
    Example 1:
    Replace the function's arguments with `i` and `j` to get another call
    `foo(i, j)`.
    '''
    param_dict = {
        'a': 'i',
        'b': 'j',
    }
    new_rt1 = rt1.replace(param_dict)
    print(new_rt1.to_str())
    # print(new_rt1.to_indented_repr())
    # print(new_rt1.get_key2spans())

    '''
    Example 2:
    Suppose that the pseduo function call `foo(a, b)` will return a pair
    `<a,b>`, and we have variable x = "xxx" and z = "zzz". Now we want to get
    the result of `foo(x, foo("yyy", z))`, which is `<"xxx",<"yyy","zzz">>`.
    '''
    content = 'foo(x, foo("yyy", z))'
    key2spans = {
        'foo': [(0, 21), (7, 20)],
        'x': [(4, 5)],
        '"yyy"': [(11, 16)],
        'z': [(18, 19)],
    }
    rt2 = ReplacerFactory.create_tree(content, key2spans)

    def foo_handler(node: ReplacerNode) -> ReplacerNode:
        remainder = ['<'] + [','] * (len(node.children) - 1) + ['>']
        return ReplacerNode(None, remainder, node.children)

    handler_dict = {
        'foo': foo_handler,
        'x': '"xxx"',
        'z': '"zzz"',
    }
    result_rt2 = rt2.replace(handler_dict)
    print(result_rt2.to_str())
    # print(result_rt2.to_indented_repr())
    # print(result_rt2.get_key2spans())

