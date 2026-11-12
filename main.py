import numpy as np
from dataclasses import dataclass

ALL_RULES_PRIHOD = ["fast_device", "short_queue"]
ALL_RULES_PEREHOD = ["perehod", "no_perehod"]


RULE_PRIHOD = ALL_RULES_PRIHOD[1]
RULE_PEREHOD = ALL_RULES_PEREHOD[0]

@dataclass
class Params:
    lam: float      #Интенсивность входящего потока
    mu1: float      #Интенсивность обслуживания прибора 1
    mu2: float      #Интенсивность обслуживания прибора2
    d: int               #порог: если |n1-n2| > d, одно требование перебрасывается
    i: int = 2
    j: int = 2


def build_states(p:Params):
    '''Перечисляем все состояния (n1, s1, n2, s2)'''


    states = [] #нужно для поиска состояния по индексу
    index = {} #нужно для поиска индекса по состоянию

    for n1 in range(p.i+1):
        for n2 in range(p.j+1):
            if RULE_PEREHOD == "perehod" and ( n1-n2 > p.d or n2-n1 > p.d ):
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
    if(RULE_PEREHOD == "perehod"):
        n1, n2 = state

        if n1 - n2 > d:
            n1 -= 1
            n2 += 1
        elif n2-n1 > d:
            n2 -= 1
            n1 += 1

        return (n1, n2)

    elif(RULE_PEREHOD == "no_perehod"):
        return state


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
                new_state = apply_rebalance((cn1, cn2), p.d)
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
        "L1_channel":0.0, "L2_channel":0.0,
        "L_queue": 0.0,"L_channel": 0.0,
        "rho1": 0.0, "rho2": 0.0,
        "P_both_busy": 0.0, "P_empty": 0.0,
        "P_imbalance": 0.0,
        "W_queue":0.0, "W_total":0.0,
    }
    for k, (n1, n2) in enumerate(states):
        w = pi[k]
        m["L1_queue"] += w * (n1-1) if n1-1 > 0 else 0
        m["L2_queue"] += w * (n2-1) if n2-1 > 0 else 0
        m["L1_channel"] += w * n1
        m["L2_channel"] += w * n2
        m["rho1"] += w * 1 if n1>0 else 0
        m["rho2"] += w * 1 if n2>0 else 0
        if n1 > 0 and n2 > 0:
            m["P_both_busy"] += w
        if n1 == 0  and n2 == 0:
            m["P_empty"] += w
        if abs(n1 - n2) > p.d:
            m["P_imbalance"] += w

    m["L_queue"] += m["L1_queue"] + m["L2_queue"]
    m["W_queue"] = m["L_queue"] / p.lam

    m["L_channel"] = m["L1_channel"] + m["L2_channel"]
    m["W_total"] = m["L_channel"]/p.lam

    return m


def experiment1():
    RULE_PEREHOD=ALL_RULES_PEREHOD[0]
    RULE_PRIHOD=ALL_RULES_PRIHOD[0]
    print("Эксперимент 1. Варьирование лямбды")
    first_lam = 10
    p = Params(lam=first_lam, mu1 = 10.0, mu2 = 10.0, d=1, i=200, j=200)

    for i in range(first_lam, 25, 1):
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

        if p.lam==first_lam:
            print('lam\t', end='')
            for k, v in m.items():
                print(f"{k:^11}", end='')
            print()
        print(i, end='\t')
        for k, v in m.items():
            print(f"{v:^11.2f}", end='')
        print()



def experiment2():
    RULE_PEREHOD=ALL_RULES_PEREHOD[0]
    RULE_PRIHOD=ALL_RULES_PRIHOD[0]
    print("Эксперимент 2. Варьирование порогового значения d")
    first_d = 1
    p = Params(lam=15, mu1 = 10.0, mu2 = 11.0, d=first_d, i=200, j=200)

    for i in range(first_d, 15, 1):
        p.d=i
        states, index = build_states(p)
        #print(f"Число состояний: {len(states)}")

        Q = build_generator(p, states, index)
        #print(f"Размер Q: {Q.shape}")
        #print(f"Макс. |сумма строки|: {np.abs(Q.sum(axis=1)).max():.2e}")

        pi = solve_stationary(Q)
        #print(f"Σπ = {pi.sum():.6f},  min π = {pi.min():.2e}")

        m = compute_metrics(pi, states, p)
        #print("\n=== Характеристики ===")

        if p.d==first_d:
            print('d\t', end='')
            for k, v in m.items():
                print(f"{k:^11}", end='')
            print()
        print(i, end='\t')
        for k, v in m.items():
            print(f"{v:^11.2f}", end='')
        print()


def experiment3():
    RULE_PEREHOD=ALL_RULES_PEREHOD[1]
    RULE_PRIHOD=ALL_RULES_PRIHOD[0]
    print("Эксперимент 3. Система без переходов требований")
    first_lam = 10
    p = Params(lam=first_lam, mu1 = 5.0, mu2 = 15.0, d=1, i=20, j=20)

    for i in range(first_lam, 25, 1):
        p.lam=i
        states, index = build_states(p)
        Q = build_generator(p, states, index)
        pi = solve_stationary(Q)
        m = compute_metrics(pi, states, p)

        if p.lam==first_lam:
            print('lam\t', end='')
            for k, v in m.items():
                print(f"{k:^11}", end='')
            print()
        print(i, end='\t')
        for k, v in m.items():
            print(f"{v:^11.2f}", end='')
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
        print("по правилу быстрейшего прибора", end=' ')
    elif RULE_PRIHOD == "short_queue":
        print("по правилу кратчайшей очереди", end=' ')

    if RULE_PEREHOD == "perehod":
        print("с переходами")
        m = compute_metrics(pi, states, p)
    elif RULE_PEREHOD == "no_perehod":
        print("без переходов")
        m = compute_metrics(pi, states, p)

    for i in range(len(states)):
        print(f'P{states[i]}: {pi[i]:.6f}')


    #experiment1()
    #experiment2()
    experiment3()


























