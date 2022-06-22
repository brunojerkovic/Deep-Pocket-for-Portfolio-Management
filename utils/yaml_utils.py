import functools
import yaml


def define_yaml_join_operator():
    class YAMLJoinOperator(yaml.YAMLObject):
        yaml_loader = yaml.SafeLoader
        yaml_tag = '!join'

        @classmethod
        def from_yaml(cls, loader, node):
            return functools.reduce(lambda a, b: a.value + b.value, node.value)
