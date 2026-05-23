from __future__ import annotations

from starlette.requests import HTTPConnection

from mlp_observatory.app.container import Container


def get_container(connection: HTTPConnection) -> Container:
    return connection.app.state.container
