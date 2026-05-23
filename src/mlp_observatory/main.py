from __future__ import annotations

import uvicorn

from mlp_observatory.app.factory import create_app
from mlp_observatory.core.settings import settings


app = create_app()


def main() -> None:
    uvicorn.run(
        "mlp_observatory.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        factory=False,
    )


if __name__ == "__main__":
    main()
