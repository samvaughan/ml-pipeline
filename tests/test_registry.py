from mlpipe import Registry
import pytest


def test_registry_create():
    reg = Registry("transformer")

    # Just a stub
    @reg.register("my_thing")
    class MyThing:
        def __init__(self, value):
            self.value = value

    instance = reg.create("my_thing", value=42)
    assert isinstance(instance, MyThing)
    assert instance.value == 42


def test_registry_unknown_name_raises():
    reg = Registry("transformer")
    with pytest.raises(KeyError, match="Unknown transformer"):
        reg.create("does_not_exist")


def test_registry_duplicate_raises():
    reg = Registry("transformer")

    @reg.register("foo")
    class A:
        pass

    with pytest.raises(KeyError, match="already registered"):

        @reg.register("foo")
        class B:
            pass
