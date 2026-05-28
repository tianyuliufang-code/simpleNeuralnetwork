scorecard = []
for score in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    print(corrent_label,"correct label")
    inputs = (np.asfarray(all_values[1:])/255.0*0.99)+0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    print(label,"network's answer")
    if(label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)

    pass