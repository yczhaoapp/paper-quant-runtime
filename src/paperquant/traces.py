"""Shared checks for the current complete-fill order and action contract."""

from decimal import Decimal

from paperquant.models import RunReport, SubmitOrder, TargetPosition


def validate_traces(report: RunReport) -> None:
    plan = report.plan
    accepted = {}
    terminal = {}
    for order in report.orders:
        if order.symbol not in plan.symbols:
            raise ValueError("Order has an unknown symbol")
        if order.status == "accepted":
            if order.order_id in accepted or order.order_id in terminal:
                raise ValueError("Order ID has multiple creation events")
            accepted[order.order_id] = order
        else:
            creation = accepted.get(order.order_id)
            if (creation is None or order.order_id in terminal
                or creation.symbol != order.symbol or order.timestamp < creation.timestamp):
                raise ValueError("Order has an invalid lifecycle transition")
            terminal[order.order_id] = order
    if accepted.keys() != terminal.keys():
        raise ValueError("Order lifecycle is incomplete")
    filled_ids = {key for key, order in terminal.items() if order.status == "filled"}
    seen_fills: set[str] = set()
    quantities: dict[str, Decimal] = {}
    for fill in report.fills:
        if fill.order_id in seen_fills:
            raise ValueError("Multiple fills for an order: partial fills are unsupported")
        seen_fills.add(fill.order_id)
        creation = accepted.get(fill.order_id)
        completion = terminal.get(fill.order_id)
        if (creation is None or completion is None or completion.status != "filled"
            or fill.symbol != creation.symbol or fill.timestamp != completion.timestamp
            or fill.timestamp < creation.timestamp):
            raise ValueError("Fill differs from its order lifecycle")
        if plan.max_order_quantity is not None and fill.quantity > plan.max_order_quantity:
            raise ValueError("Fill exceeds the declared order quantity limit")
        quantity = quantities.get(fill.symbol, Decimal(0))
        quantity += fill.quantity if fill.side == "buy" else -fill.quantity
        quantities[fill.symbol] = quantity
        if plan.max_abs_position is not None and abs(quantity) > plan.max_abs_position:
            raise ValueError("Fill exceeds the declared position limit")
    if seen_fills != filled_ids:
        raise ValueError("Completed order has no matching fill")
    for decision in report.decisions:
        targets: set[str] = set()
        for action in decision.actions:
            if action.kind not in plan.allowed_actions:
                raise ValueError("Decision contains an undeclared action")
            if hasattr(action, "symbol") and action.symbol not in plan.symbols:
                raise ValueError("Decision contains an unknown symbol")
            if isinstance(action, TargetPosition):
                if action.symbol in targets:
                    raise ValueError("Decision contains duplicate target intents")
                targets.add(action.symbol)
                if (plan.max_abs_position is not None
                    and abs(action.quantity) > plan.max_abs_position):
                    raise ValueError("Target exceeds the declared position limit")
            elif isinstance(action, SubmitOrder):
                if (plan.max_order_quantity is not None
                    and action.quantity > plan.max_order_quantity):
                    raise ValueError("Order exceeds the declared quantity limit")
    for account in report.accounts:
        if len(account.open_order_ids) != len(set(account.open_order_ids)):
            raise ValueError("Account contains duplicate open order IDs")
        for order_id in account.open_order_ids:
            creation = accepted.get(order_id)
            # Account snapshots follow event availability order. A later-visited
            # symbol's bar may fill at its earlier open, so terminal wall time
            # cannot determine whether it was pending at this snapshot.
            if creation is None or creation.timestamp > account.timestamp:
                raise ValueError("Account contains an invalid open order ID")
        if len(account.positions) != len({position.symbol for position in account.positions}):
            raise ValueError("Account contains duplicate positions")
        for position in account.positions:
            if position.symbol not in plan.symbols:
                raise ValueError("Account position has an unknown symbol")
            if (plan.max_abs_position is not None
                and abs(position.quantity) > plan.max_abs_position):
                raise ValueError("Account exceeds the declared position limit")
