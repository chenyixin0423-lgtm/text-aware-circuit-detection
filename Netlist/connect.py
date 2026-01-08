def connect(pairs, map_id_to_name):
    try:
        names = {
            0: "pmos",
            1: "nmos",
            2: "NPN",
            3: "PNP",
            4: "indu",
            5: "Diode",
            6: "R",
            7: "C",
            8: "ground",
            9: "V",
            10: "I",
        }
        type_name = {
            "pmos": "M",
            "nmos": "M",
            "NPN": "Q",
            "PNP": "Q",
            "indu": "L",
            "Diode": "D",
            "R": "R",
            "C": "C",
            "ground": "G",
            "V": "V",
            "I": "I",
        }
        total_nets = 0
        unique_name = {}  # id : "M0"
        type_num = {}  # "M" :

        nets_info = {}  # id :[]

        for i in names.values():
            type_num[type_name[i]] = 0
        for i in pairs:
            a = i[0]
            b = i[1]
            if a not in unique_name:
                unique_name[a] = type_name[map_id_to_name[a]] + str(
                    type_num[type_name[map_id_to_name[a]]]
                )
                type_num[type_name[map_id_to_name[a]]] += 1

            if b not in unique_name:
                unique_name[b] = type_name[map_id_to_name[b]] + str(
                    type_num[type_name[map_id_to_name[b]]]
                )
                type_num[type_name[map_id_to_name[b]]] += 1

        net_id = 0
        for i in unique_name:
            nets_info[i] = [[], 0]
            if unique_name[i].startswith("R"):
                nets_info[i][0].append(net_id)
                nets_info[i][0].append(net_id + 1)
                net_id += 2
            elif unique_name[i].startswith("C"):
                nets_info[i][0].append(net_id)
                nets_info[i][0].append(net_id + 1)
                net_id += 2
            elif unique_name[i].startswith("L"):
                nets_info[i][0].append(net_id)
                nets_info[i][0].append(net_id + 1)
                net_id += 2
            elif unique_name[i].startswith("M"):
                nets_info[i][0].append(net_id)
                nets_info[i][0].append(net_id + 1)
                nets_info[i][0].append(net_id + 2)
                net_id += 3
            elif unique_name[i].startswith("G"):
                nets_info[i][0].append(net_id)
                nets_info[i][0].append(net_id + 1)
                nets_info[i][0].append(net_id + 2)
                nets_info[i][0].append(net_id + 3)
                nets_info[i][0].append(net_id + 4)
                nets_info[i][0].append(net_id + 5)
                nets_info[i][0].append(net_id + 6)
                nets_info[i][0].append(net_id + 7)
                net_id += 8
            elif unique_name[i].startswith("I"):
                nets_info[i][0].append(net_id)
                nets_info[i][0].append(net_id + 1)
                net_id += 2
            elif unique_name[i].startswith("V"):
                nets_info[i][0].append(net_id)
                nets_info[i][0].append(net_id + 1)
                net_id += 2
            elif unique_name[i].startswith("D"):
                nets_info[i][0].append(net_id)
                nets_info[i][0].append(net_id + 1)
                net_id += 2
            elif unique_name[i].startswith("Q"):
                nets_info[i][0].append(net_id)
                nets_info[i][0].append(net_id + 1)
                nets_info[i][0].append(net_id + 2)
                net_id += 3

        print(nets_info, "nets_info", pairs)
        for i in pairs:
            try:
                a = i[0]
                b = i[1]
                assignid = net_id
                net_id += 1
                print(assignid, a, b)
                print(nets_info[a][1])
                print(nets_info[b][1])
                print(nets_info[a][0][nets_info[a][1]])
                print(nets_info[b][0][nets_info[b][1]])
                print(
                    nets_info[a][0][nets_info[a][1]],
                    nets_info[b][0][nets_info[b][1]],
                    assignid,
                )
                nets_info[a][0][nets_info[a][1]] = assignid
                nets_info[b][0][nets_info[b][1]] = assignid

                nets_info[a][1] = 1 + nets_info[a][1]
                nets_info[b][1] = 1 + nets_info[b][1]
            except Exception as e:
                continue
        print(unique_name)
        print(type_num)

        groundId = -1

        print(nets_info)

        for i in nets_info:
            if unique_name[i].startswith("G"):
                groundId = nets_info[i][0][0]

        strIng = ""

        for i in nets_info:
            id = i
            nets = nets_info[i][0]
            name = unique_name[id]
            print(name, end=" ")
            strIng += name + " "
            print("(", end=" ")
            strIng += "(" + " "
            for j in nets:
                print("net" + str(j), end=" ")
                strIng += "net" + str(j) + " "
            if name.startswith("M"):

                # currentId= nets[2]
                # for j in nets_info:

                strIng += "net" + str(groundId) + " "
                print("net" + str(groundId), end=" ")
            print(")", end=" ")
            strIng += ")" + " "
            print(map_id_to_name[id])
            strIng += map_id_to_name[id] + "\n"
        return strIng
    except Exception as e:
        import traceback
        print(e)
        print(traceback.format_exc())
        return str(e)+traceback.format_exc()
