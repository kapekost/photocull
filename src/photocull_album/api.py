"""HTTP surface for the sequencing UI. Every mutating endpoint validates uuids against
the keeper pool this launch was built with -- the same "never trust client input past a
lookup" posture `photocull_review`'s decisions endpoints use, applied here because this
app is about to hand a browser tab the power to enqueue a real iCloud export."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from photocull.album_store import AlbumStore
from photocull.albums import TripGroup
from photocull.printlayout import PRINT_SIZES, compute_crop_rect

security = HTTPBearer()


class CreateAlbumRequest(BaseModel):
    title: str
    print_size: str
    uuids: list[str]


class ReorderRequest(BaseModel):
    uuids: list[str]


class CropOffsetRequest(BaseModel):
    offset_x: float
    offset_y: float


def build_app(
    *,
    pool_records: list[Any],
    trip_groups: list[TripGroup],
    store: AlbumStore,
    token: str,
) -> FastAPI:
    app = FastAPI()
    pool_uuids = {r.uuid for r in pool_records}
    pool_by_uuid = {r.uuid: r for r in pool_records}

    def _check_token(creds: HTTPAuthorizationCredentials = Depends(security)) -> None:
        if creds.credentials != token:
            raise HTTPException(status_code=401, detail="invalid token")

    @app.get("/api/trips", dependencies=[Depends(_check_token)])
    def list_trips():
        return [
            {
                "title": g.title,
                "count": len(g),
                "wide_spread": g.wide_spread,
                "uuids": [r.uuid for r in g.records],
            }
            for g in trip_groups
        ]

    @app.post("/api/albums", dependencies=[Depends(_check_token)])
    def create_album(body: CreateAlbumRequest):
        unknown = [u for u in body.uuids if u not in pool_uuids]
        if unknown:
            raise HTTPException(status_code=400, detail=f"uuids not in the keeper pool: {unknown}")
        try:
            record = store.create(body.title, body.print_size, body.uuids)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"album_id": record.album_id}

    @app.put("/api/albums/{album_id}/sequence", dependencies=[Depends(_check_token)])
    def reorder(album_id: str, body: ReorderRequest):
        unknown = [u for u in body.uuids if u not in pool_uuids]
        if unknown:
            raise HTTPException(status_code=400, detail=f"uuids not in the keeper pool: {unknown}")
        try:
            store.save_sequence(album_id, body.uuids)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown album") from None
        return {"ok": True}

    @app.get("/api/albums/{album_id}", dependencies=[Depends(_check_token)])
    def get_album(album_id: str):
        record = store.get(album_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown album")
        return {
            "album_id": record.album_id,
            "title": record.title,
            "print_size": record.print_size,
            "sequence": list(record.sequence),
        }

    @app.get("/api/crop-preview/{uid}", dependencies=[Depends(_check_token)])
    def crop_preview(uid: str, print_size: str):
        record = pool_by_uuid.get(uid)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown uuid")
        if print_size not in PRINT_SIZES:
            raise HTTPException(status_code=400, detail=f"unknown print size {print_size!r}")
        size = PRINT_SIZES[print_size]
        crop = compute_crop_rect(record.width, record.height, size.width_px, size.height_px)
        needs_crop = crop.width < record.width or crop.height < record.height
        return {"needs_crop": needs_crop, "offset_x": 0.5, "offset_y": 0.5}

    @app.put("/api/albums/{album_id}/crop/{uid}", dependencies=[Depends(_check_token)])
    def set_crop(album_id: str, uid: str, body: CropOffsetRequest):
        if uid not in pool_uuids:
            raise HTTPException(status_code=400, detail=f"uuid not in the keeper pool: {uid}")
        try:
            store.set_crop_offset(album_id, uid, body.offset_x, body.offset_y)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown album") from None
        return {"ok": True}

    return app
