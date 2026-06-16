"""Type stub for the protobuf-generated worker_pb2 module.

worker_pb2.py builds its message classes dynamically at import time via
google.protobuf's _builder, so static type checkers cannot see them. This stub
mirrors worker.proto so pyright/IDE tooling can resolve the message types.

Hand-written to match `protoc --pyi_out` output; regenerate (or update by hand)
if worker.proto changes.
"""

from collections.abc import Iterable, Mapping
from typing import ClassVar, Optional, Union

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StartContainerRequest(_message.Message):
    __slots__ = ("image", "work_dir", "env_vars", "cpu_limit", "memory_limit_mb")

    class EnvVarsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: ClassVar[int]
        VALUE_FIELD_NUMBER: ClassVar[int]
        key: str
        value: str
        def __init__(self, key: Optional[str] = ..., value: Optional[str] = ...) -> None: ...

    IMAGE_FIELD_NUMBER: ClassVar[int]
    WORK_DIR_FIELD_NUMBER: ClassVar[int]
    ENV_VARS_FIELD_NUMBER: ClassVar[int]
    CPU_LIMIT_FIELD_NUMBER: ClassVar[int]
    MEMORY_LIMIT_MB_FIELD_NUMBER: ClassVar[int]
    image: str
    work_dir: str
    env_vars: _containers.ScalarMap[str, str]
    cpu_limit: int
    memory_limit_mb: int
    def __init__(
        self,
        image: Optional[str] = ...,
        work_dir: Optional[str] = ...,
        env_vars: Optional[Mapping[str, str]] = ...,
        cpu_limit: Optional[int] = ...,
        memory_limit_mb: Optional[int] = ...,
    ) -> None: ...

class StartContainerResponse(_message.Message):
    __slots__ = ("container_id",)
    CONTAINER_ID_FIELD_NUMBER: ClassVar[int]
    container_id: str
    def __init__(self, container_id: Optional[str] = ...) -> None: ...

class StopContainerRequest(_message.Message):
    __slots__ = ("container_id",)
    CONTAINER_ID_FIELD_NUMBER: ClassVar[int]
    container_id: str
    def __init__(self, container_id: Optional[str] = ...) -> None: ...

class ExecRequest(_message.Message):
    __slots__ = ("container_id", "cmd", "timeout_sec")
    CONTAINER_ID_FIELD_NUMBER: ClassVar[int]
    CMD_FIELD_NUMBER: ClassVar[int]
    TIMEOUT_SEC_FIELD_NUMBER: ClassVar[int]
    container_id: str
    cmd: str
    timeout_sec: int
    def __init__(
        self,
        container_id: Optional[str] = ...,
        cmd: Optional[str] = ...,
        timeout_sec: Optional[int] = ...,
    ) -> None: ...

class ExecChunk(_message.Message):
    __slots__ = ("stream", "data", "exit_code")

    class Stream(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STDOUT: _ExecChunk_Stream_ValueType
        STDERR: _ExecChunk_Stream_ValueType
        EXIT: _ExecChunk_Stream_ValueType

    STDOUT: ExecChunk.Stream
    STDERR: ExecChunk.Stream
    EXIT: ExecChunk.Stream
    STREAM_FIELD_NUMBER: ClassVar[int]
    DATA_FIELD_NUMBER: ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: ClassVar[int]
    stream: ExecChunk.Stream
    data: bytes
    exit_code: int
    def __init__(
        self,
        stream: Optional[Union[ExecChunk.Stream, str]] = ...,
        data: Optional[bytes] = ...,
        exit_code: Optional[int] = ...,
    ) -> None: ...

_ExecChunk_Stream_ValueType = int

class WriteFileRequest(_message.Message):
    __slots__ = ("container_id", "path", "content", "mode")
    CONTAINER_ID_FIELD_NUMBER: ClassVar[int]
    PATH_FIELD_NUMBER: ClassVar[int]
    CONTENT_FIELD_NUMBER: ClassVar[int]
    MODE_FIELD_NUMBER: ClassVar[int]
    container_id: str
    path: str
    content: bytes
    mode: int
    def __init__(
        self,
        container_id: Optional[str] = ...,
        path: Optional[str] = ...,
        content: Optional[bytes] = ...,
        mode: Optional[int] = ...,
    ) -> None: ...

class ReadFileRequest(_message.Message):
    __slots__ = ("container_id", "path")
    CONTAINER_ID_FIELD_NUMBER: ClassVar[int]
    PATH_FIELD_NUMBER: ClassVar[int]
    container_id: str
    path: str
    def __init__(self, container_id: Optional[str] = ..., path: Optional[str] = ...) -> None: ...

class FileContent(_message.Message):
    __slots__ = ("content", "size")
    CONTENT_FIELD_NUMBER: ClassVar[int]
    SIZE_FIELD_NUMBER: ClassVar[int]
    content: bytes
    size: int
    def __init__(self, content: Optional[bytes] = ..., size: Optional[int] = ...) -> None: ...

class ListDirRequest(_message.Message):
    __slots__ = ("container_id", "path")
    CONTAINER_ID_FIELD_NUMBER: ClassVar[int]
    PATH_FIELD_NUMBER: ClassVar[int]
    container_id: str
    path: str
    def __init__(self, container_id: Optional[str] = ..., path: Optional[str] = ...) -> None: ...

class DirEntry(_message.Message):
    __slots__ = ("name", "is_dir", "size")
    NAME_FIELD_NUMBER: ClassVar[int]
    IS_DIR_FIELD_NUMBER: ClassVar[int]
    SIZE_FIELD_NUMBER: ClassVar[int]
    name: str
    is_dir: bool
    size: int
    def __init__(
        self,
        name: Optional[str] = ...,
        is_dir: bool = ...,
        size: Optional[int] = ...,
    ) -> None: ...

class DirListing(_message.Message):
    __slots__ = ("entries",)
    ENTRIES_FIELD_NUMBER: ClassVar[int]
    entries: _containers.RepeatedCompositeFieldContainer[DirEntry]
    def __init__(self, entries: Optional[Iterable[Union[DirEntry, Mapping]]] = ...) -> None: ...

class HealthStatus(_message.Message):
    __slots__ = ("healthy", "active_containers", "version")
    HEALTHY_FIELD_NUMBER: ClassVar[int]
    ACTIVE_CONTAINERS_FIELD_NUMBER: ClassVar[int]
    VERSION_FIELD_NUMBER: ClassVar[int]
    healthy: bool
    active_containers: int
    version: str
    def __init__(
        self,
        healthy: bool = ...,
        active_containers: Optional[int] = ...,
        version: Optional[str] = ...,
    ) -> None: ...
