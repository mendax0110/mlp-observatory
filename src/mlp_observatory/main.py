from __future__ import annotations

import uvicorn

from mlp_observatory.app.factory import create_app
from mlp_observatory.core.settings import settings
from mlp_observatory.core.logging import configure_logging

#app = create_app()


def main() -> None:
    configure_logging()
    uvicorn.run(
        "mlp_observatory.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        factory=False,
        log_config=None,
    )
    
def create_app_for_uvicorn():
    return create_app()

app = create_app_for_uvicorn() 


if __name__ == "__main__":
    main()
