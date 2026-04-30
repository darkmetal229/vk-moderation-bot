import logging
import aiohttp
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)
VK_API_BASE = "https://api.vk.com/method"

class VKClient:
    def __init__(self):
        self.token = settings.vk_group_token
        self.user_token = settings.vk_user_token
        self.version = settings.vk_api_version
        self.group_id = settings.vk_group_id
        self.admin_id = settings.vk_admin_user_id

    async def _call(self, method: str, params: dict, use_user_token: bool = False) -> Optional[dict]:
        token = self.user_token if use_user_token else self.token
        params.update({"access_token": token, "v": self.version})
        url = f"{VK_API_BASE}/{method}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=params) as resp:
                    data = await resp.json()
                    if "error" in data:
                        err = data["error"]
                        logger.error("VK API error [%s]: code=%s msg=%s", method,
                                     err.get("error_code"), err.get("error_msg"))
                        return None
                    return data.get("response")
        except Exception as e:
            logger.error("VK API request failed [%s]: %s", method, e)
            return None

    async def notify_admin(self, message: str) -> bool:
        result = await self._call("messages.send", {
            "user_id": self.admin_id,
            "message": message,
            "random_id": 0,
        })
        return result is not None

    async def get_user_info(self, user_id: int) -> Optional[dict]:
        result = await self._call("users.get", {
            "user_ids": user_id,
            "fields": "first_name,last_name",
        })
        if result and len(result) > 0:
            u = result[0]
            return {"name": f"{u.get('first_name', '')} {u.get('last_name', '')}".strip()}
        return None

    async def delete_comment(self, owner_id: int, comment_id: int) -> bool:
        result = await self._call("wall.deleteComment", {
            "owner_id": owner_id,
            "comment_id": comment_id,
        }, use_user_token=True)

        if result == 1:
            logger.info(f"✅ Комментарий {comment_id} удалён из ВК")
            return True
        else:
            logger.warning(f"⚠️ wall.deleteComment вернул: {result}")
            return False

_client: Optional[VKClient] = None

def get_vk_client() -> VKClient:
    global _client
    if _client is None:
        _client = VKClient()
    return _client
