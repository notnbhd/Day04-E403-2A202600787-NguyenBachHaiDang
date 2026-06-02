from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from src.utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    """
    Build a detailed, strict system prompt that enforces the correct tool
    ordering, clarification rules, guardrails, and grounded Vietnamese answers.
    """
    current_day = today or "2026-06-01"
    return f"""You are an electronics order assistant. Today is {current_day}. Always reply in Vietnamese.

HARD RULES (no exceptions):
- Never call any tool in CASE A or CASE B.
- Do not infer missing fields. If a field is not explicit, treat it as missing.
- Required fields: customer name, phone number, email, shipping address, and at least 1 product with quantity.
- If any required field is missing, ask ONLY for the missing fields and stop.

STEP 0 — CLASSIFY the user request into exactly one case:

CASE A — POLICY VIOLATION (bypass stock, fake discount, fake invoice, ignore catalog/policy):
→ Refuse politely in Vietnamese. Do NOT call any tool.

CASE B — MISSING INFO (missing any required field):
→ Ask for the missing fields in Vietnamese. Do NOT call any tool. Do NOT confirm an order.

CASE C — COMPLETE ORDER REQUEST (all required fields present, no policy violation):
→ Start calling tools IMMEDIATELY. Do NOT ask for confirmation. Default quantity is 1 if not stated.

TOOL SEQUENCE FOR CASE C (follow exactly, do NOT skip any step):

STEP 1: Call `list_products` with query containing ALL product names from the request in ONE call. Example: query="ASUS ROG Zephyrus G14 Logitech Pebble 2 M350s LG UltraGear 27GP850-B". Set limit=20. Do NOT call list_products multiple times.

STEP 2: Call `get_product_details` with ALL product_ids found in step 1. Then CHECK STOCK: if any product's stock < requested quantity → STOP IMMEDIATELY, tell customer which item has insufficient stock, and ask to adjust the quantity. Do NOT call `get_discount`, `calculate_order_totals`, or `save_order`.

STEP 3: Call `get_discount` with seed_hint = customer email.

STEP 4: Call `calculate_order_totals` with items, detail_token from step 2, discount_rate from step 3. If status="error" → STOP, report error.

STEP 5: Call `save_order` with ALL customer info and order data. You MUST call this after step 4 succeeds. Do NOT stop after step 4.

GROUNDING RULES:
- Use ONLY data returned by tools. Never invent product_id, price, discount, totals, order_id, or file path.
- Use product names and quantities from tool outputs.
- Pass detail_token, discount_rate, campaign_code EXACTLY as received.

FINAL ANSWER:
- CASE A: short refusal.
- CASE B: short clarification listing only missing fields.
- CASE C success:
  - Respond in Vietnamese with a one-line confirmation.
  - If the user mixes English and Vietnamese, add a brief acknowledgement like "Đã hiểu yêu cầu song ngữ."
  - Then output a compact VALID JSON object with keys: order_id, customer (name, phone, email, shipping_address), items (name, quantity), discount_rate, final_total, save_path.
""".strip()


def build_tools(store: OrderDataStore):
    """
    Define exactly five tools with strong Pydantic schemas and compact outputs.
    Each tool delegates to the OrderDataStore for deterministic behavior.
    """

    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the product catalog. Put ALL product names in the query field separated by spaces. Call this tool ONLY ONCE with all products. Set limit=20."""
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags or [],
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Get exact product details (price, stock, SKU, warranty) for specific product IDs discovered by list_products.
        Returns a detail_token that you MUST pass to calculate_order_totals and save_order for validation.
        Check the stock field: if requested quantity exceeds available stock, do NOT proceed to save_order.
        """
        return json.dumps(store.get_product_details(product_ids), ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Get the campaign discount for this order. Use the customer's email as seed_hint.
        Returns discount_rate and campaign_code that you MUST pass to calculate_order_totals and save_order.
        Do NOT invent or override the discount rate. Only use 'standard' or 'vip' for customer_tier.
        """
        return json.dumps(store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier), ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items: list[OrderLineInput], detail_token: str, discount_rate: float) -> str:
        """Validate stock availability and calculate the discounted order total.
        Requires the detail_token from get_product_details and discount_rate from get_discount.
        Items must use exact product_id values and quantities.
        If status is 'error', do NOT call save_order. Report the error to the customer.
        """
        payload = store.calculate_order_totals(
            items=items,
            detail_token=detail_token,
            discount_rate=discount_rate,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items: list[OrderLineInput],
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the validated order to a JSON file. Only call this AFTER calculate_order_totals returns status 'ok'.
        Pass all values exactly as received from previous tool outputs:
        - detail_token from get_product_details
        - discount_rate and campaign_code from get_discount
        - items with exact product_id and quantity
        """
        result = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=items,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(result, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    """
    Create the LangGraph agent with the OrderDataStore, chat model, and tools.
    """
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    """
    Build the agent, invoke it with the user query, extract the final answer,
    tool trace, and saved order payload, and return an AgentResult.
    """
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    """Return the last non-empty AI answer from the message list."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    """Convert tool calls and tool results into a simple grading trace."""
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    """Parse the save_order tool output into (saved_order, path)."""
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None
