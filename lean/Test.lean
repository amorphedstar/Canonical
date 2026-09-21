import Canonical

example : Nat := by canonical

example : True := by canonical

example : 0 = 0 := by canonical

example (n : Nat) : n = n := by canonical

example (p : Prop) (hp : p) : p := by canonical

example (n : Nat) : n + 0 = n := by canonical 10 [Nat.add_zero]

example (n m : Nat) : n + m = m + n := by canonical 15 [Nat.add_comm]

-- Unguided: no lemma list, 8s timeout, so search has time to adopt the model.
example (n m : Nat) : n + (m + 0) = (n + m) := by canonical 8
