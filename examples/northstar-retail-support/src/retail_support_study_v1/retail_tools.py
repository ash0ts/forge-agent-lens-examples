from __future__ import annotations

from copy import deepcopy
from functools import wraps
from typing import Any

from tau2.domains.retail.data_model import Order, RetailDB
from tau2.domains.retail.tools import RetailTools
from tau2.environment.toolkit import ToolType, is_tool

TAU3_COMPATIBILITY_FIXES = (
    {
        "id": "tau3-v1.0.1-multi-item-variant-binding",
        "tool": "modify_pending_order_items",
        "upstream_version": "1.0.1",
        "behavior": (
            "bind each replacement item's price and options to its own variant "
            "instead of reusing the final validated variant"
        ),
    },
)


class StudyRetailTools(RetailTools):
    """Pinned tau3 retail tools plus reviewed, digest-bound compatibility fixes."""

    db: RetailDB

    @is_tool(ToolType.WRITE)
    @wraps(RetailTools.modify_pending_order_items)
    def modify_pending_order_items(
        self,
        order_id: str,
        item_ids: list[str],
        new_item_ids: list[str],
        payment_method_id: str,
    ) -> Order:
        """Apply the pinned upstream mutation, then repair its multi-item binding.

        tau3 v1.0.1 validates every requested replacement correctly, but its
        mutation loop reuses the final loop's ``variant`` for every changed
        item. Capture the exact target objects before the upstream call, then
        restore each target's own reviewed variant afterward. Payment, status,
        validation, and error behavior remain upstream-owned.
        """

        order = self._get_order(order_id)
        remaining = list(order.items)
        targets: list[tuple[Any, str, str]] = []
        for item_id, new_item_id in zip(item_ids, new_item_ids, strict=False):
            target = next(
                (item for item in remaining if item.item_id == item_id),
                None,
            )
            if target is None:
                # Let the pinned implementation return its normal public error.
                break
            remaining.remove(target)
            targets.append((target, target.product_id, new_item_id))

        result = super().modify_pending_order_items(
            order_id=order_id,
            item_ids=item_ids,
            new_item_ids=new_item_ids,
            payment_method_id=payment_method_id,
        )

        if len(targets) != len(item_ids):
            raise RuntimeError("upstream mutation succeeded without captured targets")
        for target, product_id, new_item_id in targets:
            variant = self._get_variant(product_id, new_item_id)
            if target.item_id != new_item_id or target.product_id != product_id:
                raise RuntimeError("upstream mutation changed the wrong retail item")
            target.price = variant.price
            target.options = deepcopy(variant.options)
        return result
