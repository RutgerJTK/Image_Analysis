def label_go_check(f_labels):
    """
    
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from fractions import Fraction 
    total = 0
    freq_labels = {}
    fracs = []  
    frac_value = float(0)
    labs = freq_labels.keys()
    freqs = freq_labels.values()
    fracs_10 = []
 
    print("-"*80)
    print("label go check:")
    for labels in f_labels['Finding_Labels']:
        for label in labels:
            if label not in freq_labels:
                freq_labels[label] = 0
            freq_labels[label] += 1
            total += 1
    
    print(freq_labels, total)   

    for key, value in freq_labels.items():
        # print("key: ", key, ", value: ", value)
        frac_value = (int(value) / int(total)) * 100
        frac_value = round(frac_value, 3)
        frac_10 = round((int(value) / 10), 0)
        fracs_10.append(int(frac_10))
        fracs.append(frac_value)

    ## pure graph, no fractioning
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.axis('equal')
    ax.pie(freqs, labels = labs, autopct='%1.2f%%')
    plt.show()

    ## 10% bar plot - so far zijn de as titels en nummers een probleem en ik haat het. 
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    for i, v in enumerate(fracs_10):
        ax.text(v + 3, i + .25, str(v), color='red', fontweight='bold')
    langs = ['C', 'C++', 'Java', 'Python', 'PHP']
    students = [23,17,35,29,12]
    ax.bar(labs, fracs_10)
    plt.show()

    # ## fractions graph 
    # fig2 = plt.figure()
    # ax = fig2.add_axes([0,0,1,1])
    # ax.axis('equal')
    # # ax.pie(fracs, labels = labs, autopct='%1.2f%%')
    # ax.pie(fracs, labels = labs)
    # plt.show()

