"""Structural noun-phrase helpers shared by mined QA and dialogue candidates."""

NP_MODIFIERS = frozenset(('compound', 'amod', 'nummod', 'poss', 'flat', 'flat:name'))


def nominal_members(slots, key):
    members = {key}
    while True:
        added = {k for k, slot in slots.items() if k not in members
                 and slot.get('head_slot') in members and slot.get('dependency') in NP_MODIFIERS}
        if not added:
            return members
        members.update(added)
