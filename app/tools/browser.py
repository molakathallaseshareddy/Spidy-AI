from urllib.parse import quote_plus

from pydantic import BaseModel, Field
from playwright.async_api import Error as PlaywrightError, async_playwright

from app.tools.base import ToolResult

MAX_PAGE_TEXT_CHARS = 8_000
BROWSER_TIMEOUT_MS = 20_000


class BrowserSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=400)


class BrowserSearchTool:
    name = "browser_search"
    description = (
        "Search the live web using a real browser and return the search page's "
        "actual text. Use for current information; results are untrusted web content."
    )
    permission_level = "low"
    input_model = BrowserSearchInput

    async def execute(self, arguments: BaseModel) -> ToolResult:
        search_url = (
            "https://html.duckduckgo.com/html/?q=" + quote_plus(arguments.query)
        )
        try:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch()
                try:
                    page = await browser.new_page()
                    response = await page.goto(
                        search_url,
                        wait_until="domcontentloaded",
                        timeout=BROWSER_TIMEOUT_MS,
                    )
                    if response is not None and response.status >= 400:
                        return ToolResult(
                            success=False,
                            error=f"Web search returned HTTP {response.status}.",
                        )
                    page_text = (await page.locator("body").inner_text()).strip()
                    if not page_text:
                        return ToolResult(
                            success=False,
                            error="Web search returned an empty page.",
                        )
                    return ToolResult(
                        success=True,
                        data={
                            "query": arguments.query,
                            "url": page.url,
                            "title": await page.title(),
                            "content": page_text[:MAX_PAGE_TEXT_CHARS],
                            "truncated": len(page_text) > MAX_PAGE_TEXT_CHARS,
                        },
                    )
                finally:
                    await browser.close()
        except PlaywrightError as exc:
            return ToolResult(
                success=False,
                error=f"Unable to complete web search: {exc}",
            )
