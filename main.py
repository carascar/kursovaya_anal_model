import numpy as np
from dataclasses import dataclass
import graph

ALL_RULES_PRIHOD = ["fast_device", "short_queue"]
ALL_RULES_PEREHOD = ["perehod", "no_perehod"]


RULE_PRIHOD = ALL_RULES_PRIHOD[0]
RULE_PEREHOD = ALL_RULES_PEREHOD[1]

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
        "rho": 0.0, "rho1": 0.0, "rho2": 0.0,
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
        m["rho"] = (m["rho1"]+m["rho2"])/2.0

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
    print("Эксперимент 1. Варьирование лямбды")
    first_lam = 12
    p = Params(lam=first_lam, mu1 = 10.0, mu2 = 10.0, d=1, i=200, j=200)
    x_values = []
    y_values_dict = {"rho":[]}


    for i in range(first_lam, 22, 1):
        p.lam=i
        states, index = build_states(p)
        Q = build_generator(p, states, index)
        pi = solve_stationary(Q)
        m = compute_metrics(pi, states, p)

        x_values.append(i)
        y_values_dict["rho"].append(m["rho"])

        if p.lam==first_lam:
            print('lam\t', end='')
            for k, v in m.items():
                print(f"{k:^11}", end='')
            print()
        print(i, end='\t')
        for k, v in m.items():
            print(f"{v:^11.2f}", end='')
        print()
    graph.plot_values(x_values, y_values_dict, title=f"Эксперимент 1(mu1={p.mu1}, mu2={p.mu2})", xlabel="лямбда",
                      ylabel="коэффициент общей загрузка системы")


def experiment2():
    print("Эксперимент 2. Варьирование порогового значения d")
    first_d = 1
    p = Params(lam=18, mu1 = 2.0, mu2 = 18.0, d=first_d, i=200, j=200)

    x_values = []
    y_values_dict = {"L_channel":[],
                     "L1_channel":[],
                     "L2_channel":[]}

    for i in range(first_d, 11, 1):
        p.d=i
        states, index = build_states(p)
        Q = build_generator(p, states, index)
        pi = solve_stationary(Q)
        m = compute_metrics(pi, states, p)


        x_values.append(i)
        #y_values.append(m["L_channel"])
        y_values_dict["L_channel"].append(m["L_channel"])
        y_values_dict["L1_channel"].append(m["L1_channel"])
        y_values_dict["L2_channel"].append(m["L2_channel"])

        if p.d==first_d:
            print('d\t', end='')
            for k, v in m.items():
                print(f"{k:^11}", end='')
            print()
        print(i, end='\t')
        for k, v in m.items():
            print(f"{v:^11.2f}", end='')
        print()
    graph.plot_values(x_values, y_values_dict, title=f"Эксперимент 2(mu1={p.mu1}, mu2={p.mu2})", xlabel="d",
                      ylabel="среднее количество требований в системе")


def experiment3():
    print("Эксперимент 3. Ассиметрия каналов")
    first_mu = 1
    p = Params(lam=18, mu1 = 10.0, mu2 = 10.0, d=1, i=200, j=200)

    x_values = []
    y_values_dict = {"L_channel":[],
                     "L1_channel":[],
                     "L2_channel":[]}

    for i in range(first_mu, 11, 1):
        p.mu1=i
        p.mu2=20-i
        states, index = build_states(p)
        Q = build_generator(p, states, index)
        pi = solve_stationary(Q)
        m = compute_metrics(pi, states, p)


        x_values.append(i)
        #y_values.append(m["L_channel"])
        y_values_dict["L_channel"].append(m["L_channel"])
        y_values_dict["L1_channel"].append(m["L1_channel"])
        y_values_dict["L2_channel"].append(m["L2_channel"])

        if p.mu1==first_mu:
            print(f'{'mu1':^11}'+f'{'mu2':^11}', end='')
            for k, v in m.items():
                print(f"{k:^11}", end='')
            print()
        print(f'{p.mu1:^11}'+f'{p.mu2:^11}',  end='')
        for k, v in m.items():
            print(f"{v:^11.2f}", end='')
        print()
    graph.plot_values(x_values, y_values_dict, title="Эксперимент 2(mu2=20-mu1)", xlabel="mu1",
                      ylabel="среднее количество требований в системе")




def experiment4():

    if(RULE_PEREHOD == "perehod"):
        print("Эксперимент 3. Система с переходом требований")
    elif(RULE_PEREHOD == "no_perehod"):
        print("Эксперимент 3. Система без переходов требований")
    first_lam = 12
    p = Params(lam=first_lam, mu1 = 1.0, mu2 = 19.0, d=1, i=50, j=50)
    x_values=[]
    y_values_dict = {"L_channel": [],
                     "L1_channel": [],
                     "L2_channel": []}

    for i in range(first_lam, 22, 1):
        p.lam=i
        states, index = build_states(p)
        Q = build_generator(p, states, index)
        pi = solve_stationary(Q)
        m = compute_metrics(pi, states, p)
        x_values.append(i)
        y_values_dict["L_channel"].append(m["L_channel"])
        y_values_dict["L1_channel"].append(m["L1_channel"])
        y_values_dict["L2_channel"].append(m["L2_channel"])

        if p.lam==first_lam:
            print('lam\t', end='')
            for k, v in m.items():
                print(f"{k:^11}", end='')
            print()
        print(i, end='\t')
        for k, v in m.items():
            print(f"{v:^11.2f}", end='')
        print()
    graph.plot_values(x_values, y_values_dict, title=f"Эксперимент 3(mu1={p.mu1}, mu2={p.mu2})", xlabel="лямбда", ylabel="среднее количество требований в системе")




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

    '''RULE_PEREHOD = "perehod"
    RULE_PRIHOD = "short_queue"
    experiment1()'''


    '''RULE_PEREHOD = "perehod"
    RULE_PRIHOD = "short_queue"
    experiment2()'''

    '''RULE_PEREHOD = "perehod"
    RULE_PRIHOD = "short_queue"
    experiment3()'''


    RULE_PEREHOD = "perehod"
    RULE_PRIHOD = "short_queue"
    experiment4()
    RULE_PEREHOD = "no_perehod"
    experiment4()

























