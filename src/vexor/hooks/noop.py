from vexor.hooks.base import VexorHook


class NoopHook(VexorHook):
    """Zero-overhead hook for production use. All methods are inherited no-ops."""
