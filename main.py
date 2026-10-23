import numpy as np
from dataclasses import dataclass

@dataclass
class Params:
    lam: float      #Интенсивность входящего потока
    mu1: float      #Интенсивность обслуживания прибора 1
    mu2: float      #Интенсивность обслуживания прибора2
    d: int               #порог: если |n1-n2| > d, одно требование перебрасывается
    N: int = 40          #усечение n1+n2<N


def build_states(p:Params):
    '''Перечисляем все состояния (n1, s1, n2, s2)'''


    states = [] #нужно для поиска состояния по индексу
    index = {} #нужно для поиска индекса по состоянию

    for n1 in range(p.N+1):
        for n2 in range(p.N+1 - n1):
            if n1-n2 > p.d or n2-n1 > p.d:
                continue
            for s1 in (0, 1):
                for s2 in (0, 1):
                    key = (n1, s1, n2, s2)
                    index[key] = len(states)
                    states.append(key)
    return states, index


def apply_rebalance(state, d):
    """
    Если |n1 - n2| > d — перебросить ОДНО ожидающее требование
    из длинной очереди в короткую. Обслуживаемые (s1, s2) не трогаем.
    """

    n1, s1, n2, s2 = state

    if n1 - n2 > d:
        if n1>0:
            n1 -= 1
            if s2==0:
                s2=1
            else:
                n2+=1
    elif n2-n1 > d:
        if n2>0:
            n2 -= 1
            if s1==0:
                s1=1
            else:
                n1+=1
    return (n1, s1, n2, s2)


def build_generator(p: Params, states, index):
    """
        Инфинитезимальный генератор Q.
        Q[i, j] — интенсивность перехода i -> j.
        Q[i, i] = -сумма ЗАПИСАННЫХ исходящих интенсивностей.
    """

    n=len(states)
    Q = np.zeros((n,n))

    for i, state in enumerate(states):
        n1, s1, n2, s2 = state
        diag=0.0

        # ---------- Приход ----------
        # Встаёт в очередь более эффективного прибора.
        if p.mu1>p.mu2:
            targets = ['q1']
        elif p.mu2>p.mu1:
            targets = ['q2']
        else:
            targets = ['q1', 'q2']

        for where in targets:
            cn1, cs1, cn2, cs2 = n1, s1, n2, s2
            if where == 'q1':
                if cs1==0:
                    cs1=1
                else:
                    cn1+=1
            else:
                if cs2==0:
                    cs2=1
                else:
                    cn2+=1
            new_state =apply_rebalance((cn1, cs1, cn2, cs2), p.d)
            if new_state in index:
                weight = p.lam/len(targets)
                Q[i, index[new_state]] += weight
                diag += weight
        # ---------- Уход с прибора 1 ----------
        if s1 == 1:
            if n1>0:
                cand = (n1-1, 1, n2, s2)
            else:
                cand = (n1, 0, n2, s2)
            cand = apply_rebalance(cand, p.d)
            if cand in index:
                Q[i, index[cand]] += p.mu1
                diag += p.mu1

        # ---------- Уход с прибора 2 ----------
        if s2 == 1:
            if n2 > 0:
                cand = (n1, s1, n2 - 1, 1)  # сразу берёт следующего
            else:
                cand = (n1, s1, n2, 0)  # прибор освобождается
            cand = apply_rebalance(cand, p.d)
            if cand in index:
                Q[i, index[cand]] += p.mu2
                diag += p.mu2

        Q[i, i] = -diag

    return Q

def solve_stationary(Q):
    """Решает πQ = 0, Σπ = 1."""
    n = Q.shape[0]
    A = Q.T.copy()
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    return pi


def compute_metrics(pi, states, p: Params):
    """Характеристики по стационарному распределению."""
    m = {
        "L1_queue": 0.0, "L2_queue": 0.0,
        "L1_total": 0.0, "L2_total": 0.0, "L_total": 0.0,
        "rho1": 0.0, "rho2": 0.0,
        "P_both_busy": 0.0, "P_empty": 0.0,
        "P_imbalance": 0.0,
        "W":0.0,
    }
    for k, (n1, s1, n2, s2) in enumerate(states):
        w = pi[k]
        m["L1_queue"] += w * n1
        m["L2_queue"] += w * n2
        m["L1_total"] += w * (n1 + s1)
        m["L2_total"] += w * (n2 + s2)
        m["rho1"] += w * s1
        m["rho2"] += w * s2
        if s1 == 1 and s2 == 1:
            m["P_both_busy"] += w
        if n1 == 0 and s1 == 0 and n2 == 0 and s2 == 0:
            m["P_empty"] += w
        if abs(n1 - n2) > p.d:
            m["P_imbalance"] += w
    m["L_total"] = m["L1_total"] + m["L2_total"]
    m["W_total"] = m["L_total"]/p.lam
    return m


# ---------------- ЗАПУСК ----------------
if __name__ == "__main__":
    p = Params(lam=190.0, mu1=100.0, mu2=100.0, d=1, N=30)

    states, index = build_states(p)
    print(f"Число состояний: {len(states)}")

    Q = build_generator(p, states, index)
    print(f"Размер Q: {Q.shape}")
    print(f"Макс. |сумма строки|: {np.abs(Q.sum(axis=1)).max():.2e}")

    pi = solve_stationary(Q)
    print(f"Σπ = {pi.sum():.6f},  min π = {pi.min():.2e}")

    m = compute_metrics(pi, states, p)
    print("\n=== Характеристики ===")
    for k, v in m.items():
        print(f"{k:>14}: {v:.6f}")

































