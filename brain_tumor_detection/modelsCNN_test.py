import matplotlib.pyplot as plt
from pandas import DataFrame

import dataset as ds
import model as md

#Analisi prestazioni con il primo modello
train_set, val_set, test_set = ds.load_data_with_validation(dataset_dir='datasets/brain_tumor_dataset', img_width=224,
                                                            img_height=224, batch_size=64)

#model1 = md.create_modelCNN1()

#result1 = model1.fit(train_set, epochs=5)

#df1 = DataFrame(result1.history).plot()
#plt.show()

#loss1, accuracy1 = model1.evaluate(test_set)
#print("loss1", loss1)
#print("accuracy1", accuracy1)

#Analisi prestazioni con il secondo modello

#model2 = md.create_modelCNN2()

#result2 = model2.fit(train_set, epochs=5)

#df2 = DataFrame(result2.history).plot()
#plt.show()

#loss2, accuracy2 = model2.evaluate(test_set)
#print("loss2", loss2)
#print("accuracy2", accuracy2)

#Analisi prestazioni con il terzo modello

model3 = md.create_modelCNN3()

result3 = model3.fit(train_set, epochs=5, validation_data=val_set)

loss3, accuracy3 = model3.evaluate(test_set)
print("loss3", loss3)
print("accuracy3", accuracy3)
