import numpy as np
from dataclasses import dataclass

ALL_RULES_PRIHOD = ["fast_device", "short_queue"]

RULE_PRIHOD = ALL_RULES_PRIHOD[1]

@dataclass
class Params:
    lam: float      #Интенсивность входящего потока
    mu1: float      #Интенсивность обслуживания прибора 1
    mu2: float      #Интенсивность обслуживания прибора2
    d: int               #порог: если |n1-n2| > d, одно требование перебрасывается
    N: int = 40          #усечение n1+n2<N
    i: int = 2
    j: int = 2


def build_states(p:Params):
    '''Перечисляем все состояния (n1, s1, n2, s2)'''


    states = [] #нужно для поиска состояния по индексу
    index = {} #нужно для поиска индекса по состоянию

    for n1 in range(p.i+1):
        for n2 in range(p.j+1):
            if n1-n2 > p.d or n2-n1 > p.d:
                continue
            key = (n1, n2)
            index[key] = len(states)
            states.append(key)
    return states, index


def apply_rebalance(state, d):
    """
    Если |n1 - n2| > d — перебросить ОДНО ожидающее требование
    из длинной очереди в короткую. Обслуживаемые (s1, s2) не трогаем.
    """

    n1, n2 = state

    if n1 - n2 > d:
        n1 -= 1
        n2 += 1
    elif n2-n1 > d:
        n2 -= 1
        n1 += 1

    return (n1, n2)


def build_generator(p: Params, states, index):
    """
        Инфинитезимальный генератор Q.
        Q[i, j] — интенсивность перехода i -> j.
        Q[i, i] = -сумма ЗАПИСАННЫХ исходящих интенсивностей.
    """

    n=len(states)
    Q = np.zeros((n,n))

    for i, state in enumerate(states):
        n1, n2 = state
        diag=0.0

        # ---------- Приход ----------
        # Встаёт в очередь более эффективного прибора.
        if RULE_PRIHOD == "fast_device":
            if p.mu1>p.mu2:
                targets = ['q1']
            elif p.mu2>p.mu1:
                targets = ['q2']
            else:
                targets = ['q1', 'q2']

            for where in targets:
                cn1, cn2 = n1, n2
                if where == 'q1':
                    cn1+=1
                else:
                    cn2+=1
                new_state =apply_rebalance((cn1, cn2), p.d)
                if new_state in index:
                    weight = p.lam/len(targets)
                    Q[i, index[new_state]] += weight
                    diag += weight

        elif RULE_PRIHOD == "short_queue":
            if n1<n2:
                targets = ['q1']
            elif n2<n1:
                targets = ['q2']
            else:
                targets = ['q1', 'q2']

            for where in targets:
                cn1, cn2 = n1, n2
                if where == 'q1':
                    cn1+=1
                else:
                    cn2+=1
                new_state =apply_rebalance((cn1, cn2), p.d)
                if new_state in index:
                    weight = p.lam/len(targets)
                    Q[i, index[new_state]] += weight
                    diag += weight


        # ---------- Уход с линии 1 ----------
        if n1>0:
            cand = (n1-1, n2)
            cand = apply_rebalance(cand, p.d)
            if cand in index:
                Q[i, index[cand]] += p.mu1
                diag += p.mu1

        # ---------- Уход с линии 2 ----------
        if n2 > 0:
            cand = (n1, n2-1) 
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
        "L_total": 0.0,
        "rho1": 0.0, "rho2": 0.0,
        "P_both_busy": 0.0, "P_empty": 0.0,
        "P_imbalance": 0.0,
        "W_total":0.0,
    }
    for k, (n1, n2) in enumerate(states):
        w = pi[k]
        m["L1_queue"] += w * n1
        m["L2_queue"] += w * n2
        m["rho1"] += w * 1 if n1>0 else 0
        m["rho2"] += w * 1 if n2>0 else 0
        if n1 > 0 and n2 > 0:
            m["P_both_busy"] += w
        if n1 == 0  and n2 == 0:
            m["P_empty"] += w
        if abs(n1 - n2) > p.d:
            m["P_imbalance"] += w
    m["L_total"] = m["L1_queue"] + m["L2_queue"]
    m["W_total"] = m["L_total"]/p.lam
    return m


def experiment1():
    p = Params(lam=10, mu1 = 100.0, mu2 = 100.0, d=1, N=30)

    for i in range(10, 200, 10):
        p.lam=i
        states, index = build_states(p)
        #print(f"Число состояний: {len(states)}")

        Q = build_generator(p, states, index)
        #print(f"Размер Q: {Q.shape}")
        #print(f"Макс. |сумма строки|: {np.abs(Q.sum(axis=1)).max():.2e}")

        pi = solve_stationary(Q)
        #print(f"Σπ = {pi.sum():.6f},  min π = {pi.min():.2e}")

        m = compute_metrics(pi, states, p)
        #print("\n=== Характеристики ===")

        if p.lam==10:
            for k, v in m.items():
                print(f"{k:>14}", end='')
            print()
        for k, v in m.items():
            print(f"{i}\t{v:>14.2f}", end='')
        print()

# ---------------- ЗАПУСК ----------------
if __name__ == "__main__":
    p = Params(lam=3, mu1=1, mu2=2, d=1, i=2, j=2)

    states, index = build_states(p)
    print(f"Число состояний: {len(states)}")

    Q = build_generator(p, states, index)
    print(f"Размер Q: {Q.shape}")
    print(f"Макс. |сумма строки|: {np.abs(Q.sum(axis=1)).max():.2e}")
    print(Q)

    pi = solve_stationary(Q)
    print(f"Σπ = {pi.sum():.6f},  min π = {pi.min():.2e}")

    m = compute_metrics(pi, states, p)
    print("\n=== Характеристики ===")
    for k, v in m.items():
        print(f"{k:>14}: {v:.6f}")

    print("\nвероятности состояний системы")
    if RULE_PRIHOD == "fast_device":
        print("по правилу быстрейшего прибора")
    elif RULE_PRIHOD == "short_queue":
        print("по правилу кратчайшей очереди")
    for i in range(len(states)):
        print(f'P{states[i]}: {pi[i]:.6f}')


    #experiment1()


























