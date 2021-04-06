import json


# bool_exp 如t , a&b,  a&b | c&d
def sat(bool_exp, state):
    # print('boolexp',bool_exp,state)
    if bool_exp == 't':
        return True
    bool_exp = bool_exp.split('|')
    for conjoin in bool_exp:
        a = conjoin.split('&')
        mark = 1
        for ap in a:
            if ap not in state:
                mark = 0
                break
        if mark == 1:
            return True
    return False


def convert_str(ori_bool_exp, APs):
    for i in range(len(APs)):
        ori_bool_exp = ori_bool_exp.replace(str(i), APs[i].replace('"', ''))
    ori_bool_exp = ori_bool_exp.replace('[', '').replace(']', '')
    return ori_bool_exp


def runAutom(APs, States, Transform, Accept, Start, trace):
    current_state = set()
    current_state.add(int(Start[0]))
    trace = trace.replace('"', '').split(',')
    trace[-1] = trace[-1].replace('{', '').replace('}', '')
    # print('current_state:', current_state)
    for i in range(0, len(trace) - 1):
        next_state = set()
        for state in current_state:  # 目前所在的每一个状态
            for key in Transform[state].keys():
                bool_exp = convert_str(key, APs)
                if sat(bool_exp, trace[i]):
                    next_state.add(int(Transform[state][key]))
            # current_state=int(Transform[current_state][tkey])
        current_state = next_state
        # print('current_state:',current_state)
    # 走到最后了

    for acc_state in Accept:  # 对每个接受态检查
        for state in current_state:  # 对每个当前状态
            if acc_state in States[state]:  # 如果这个状态是最终状态
                for key in Transform[state].keys():  # 找它转移边有无转到自身的
                    if int(Transform[state][key]) == state:  # 自身成环的转移
                        bool_exp = convert_str(key, APs)
                        if sat(bool_exp, trace[-1]):
                            return True
    return False


if __name__ == "__main__":
    f = open('random_data/0.json')
    datas = json.load(f)

    cnt = 0
    print('total len', len(datas))
    for data in datas:
        # print('data:',data)
        if data.get('trace', -1) == -1:
            continue
        if data['trace'] == "unsat":
            continue
        res = runAutom(data['APs'], data['States'], data['Transform'], data['Accept'], data['Start'], data['trace'])
        # print(res)
        if not res:
            print('cnt', cnt)
            print('data:', data)
            break
        print('\r', cnt, end='')
        cnt += 1
