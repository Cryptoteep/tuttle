from decimal import Decimal

from ..clients.intent import ClientsIntent
from ..contacts.intent import ContactsIntent
from ..core.abstractions import CrudIntent
from ..core.intent_result import IntentResult

from ...model import Client, Contract, PaymentMilestone, User, normalize_vat_rate
from ...tax import get_tax_system


class ContractsIntent(CrudIntent):
    """Handles Contract CRUD intents."""

    entity_type = Contract
    deletion_guards = [
        ("projects", "projects", lambda p: p.title),
        ("invoices", "invoices", lambda i: i.number or f"#{i.id}"),
    ]
    __save_skip__ = {"client", "projects", "invoices", "payment_milestones"}

    def __init__(self):
        super().__init__()
        self._clients_intent = ClientsIntent()
        self._contacts_intent = ContactsIntent()

    # -- Cross-entity delegates ------------------------------------------------

    def get_all_clients_as_map(self):
        return self._clients_intent.get_all_as_map()

    def get_all_contacts_as_map(self):
        return self._contacts_intent.get_all_as_map()

    def save_client(self, client: Client) -> IntentResult:
        return self._clients_intent._validated_save(client=client)

    def get_default_currency(self) -> IntentResult:
        """Derive default contract currency from the user's operating country."""
        try:
            users = self.query(User)
            country = users[0].operating_country if users else "Germany"
            ts = get_tax_system(country)
            return IntentResult(was_intent_successful=True, data=ts.currency)
        except Exception:
            return IntentResult(was_intent_successful=True, data="EUR")

    # -- Contract-specific logic -----------------------------------------------

    def _validated_save(self, contract: Contract) -> IntentResult:
        is_updating = contract.id is not None
        has_rate = contract.rate is not None and contract.rate > 0
        has_fixed = contract.fixed_price is not None and contract.fixed_price > 0
        if not has_rate and not has_fixed:
            return IntentResult(
                was_intent_successful=False,
                error_msg="A contract needs either a rate or a fixed price.",
            )
        try:
            contract.VAT_rate = normalize_vat_rate(contract.VAT_rate)
        except ValueError as e:
            return IntentResult(
                was_intent_successful=False,
                error_msg=str(e),
                log_message=f"ContractsIntent._validated_save: {e}",
            )
        result = self.save(contract)
        if not result.was_intent_successful:
            if is_updating:
                old = self.get_by_id(contract.id)
                result.data = old.data if old.was_intent_successful else None
            result.error_msg = self._describe_save_error(result.exception)
            result.log_message_if_any()
        return result

    @staticmethod
    def _describe_save_error(exc) -> str:
        if exc is None:
            return "Failed to save the contract."
        detail = str(getattr(exc, "orig", exc))
        if "UNIQUE" in detail or "duplicate" in detail.lower():
            if "title" in detail:
                return "A contract with this title already exists."
            return "A contract with these details already exists."
        if "NOT NULL" in detail:
            return "A required field is missing."
        if "FOREIGN KEY" in detail or "foreign key" in detail:
            return "The selected client is invalid."
        return "Failed to save the contract."

    toggle_complete_status = CrudIntent.toggle_completed

    # -- Milestone management --------------------------------------------------

    def save_milestones(self, contract_id, milestones) -> IntentResult:
        """Save payment milestones for a contract.

        Replaces all existing milestones with the provided list.
        Each entry is a dict with keys: title, percentage, amount, position.
        """
        result = self.get_by_id(contract_id)
        if not result.was_intent_successful or not result.data:
            return IntentResult(
                was_intent_successful=False,
                error_msg="Contract not found.",
            )
        contract = result.data

        existing_by_id = {m.id: m for m in contract.payment_milestones}
        incoming_ids = set()
        new_milestones = []

        for i, m in enumerate(milestones):
            mid = m.get("id")
            pct = m.get("percentage")
            amt = m.get("amount")
            if mid and mid in existing_by_id:
                incoming_ids.add(mid)
                ms = existing_by_id[mid]
                ms.title = m.get("title", ms.title)
                ms.percentage = Decimal(str(pct)) if pct is not None else None
                ms.amount = Decimal(str(amt)) if amt is not None else None
                ms.position = i
                new_milestones.append(ms)
            else:
                ms = PaymentMilestone(
                    contract_id=contract_id,
                    title=m.get("title", ""),
                    percentage=Decimal(str(pct)) if pct is not None else None,
                    amount=Decimal(str(amt)) if amt is not None else None,
                    position=i,
                    invoiced=False,
                )
                new_milestones.append(ms)

        if new_milestones:
            if all(ms.percentage is not None for ms in new_milestones):
                total_pct = sum(Decimal(str(ms.percentage)) for ms in new_milestones)
                if total_pct != Decimal("100"):
                    return IntentResult(
                        was_intent_successful=False,
                        error_msg=f"Milestone percentages must sum to 100% (currently {total_pct}%).",
                    )
            elif all(ms.amount is not None for ms in new_milestones):
                if contract.fixed_price is None:
                    return IntentResult(
                        was_intent_successful=False,
                        error_msg="Fixed-price contract required for amount-based milestones.",
                    )
                total_amt = sum(Decimal(str(ms.amount)) for ms in new_milestones)
                fixed = Decimal(str(contract.fixed_price))
                if total_amt != fixed:
                    return IntentResult(
                        was_intent_successful=False,
                        error_msg=(
                            f"Milestone amounts must sum to the contract fixed price "
                            f"({fixed}, currently {total_amt})."
                        ),
                    )
            else:
                return IntentResult(
                    was_intent_successful=False,
                    error_msg="Each milestone must use either percentage or amount consistently.",
                )

        # Delete removed milestones (only if not yet invoiced)
        for old_id, old_ms in existing_by_id.items():
            if old_id not in incoming_ids:
                if old_ms.invoiced:
                    return IntentResult(
                        was_intent_successful=False,
                        error_msg=f"Cannot remove milestone '{old_ms.title}' — it has already been invoiced.",
                    )
                self.delete_by_id(PaymentMilestone, old_id)

        for ms in new_milestones:
            self.store(ms)

        return IntentResult(was_intent_successful=True)

    def get_milestones(self, contract_id) -> IntentResult:
        """Get all payment milestones for a contract."""
        result = self.get_by_id(contract_id)
        if not result.was_intent_successful or not result.data:
            return IntentResult(
                was_intent_successful=False,
                error_msg="Contract not found.",
            )
        return IntentResult(
            was_intent_successful=True,
            data=result.data.payment_milestones,
        )
