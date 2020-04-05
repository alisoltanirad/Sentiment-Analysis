# Sentiment-Analysis-On-Small-Datasets
It's an open source project under MIT licence by Ali Soltani Rad.  <br/><br/>Sentiment analysis on small datasets using <b>fine-tuning</b> technique and <b>LSTM neural network</b> model.  <br/><br/>
Since the number of data records in an small dataset is not enough for a relative accurate result, I used Keras IMDB large dataset to train an LSTM neural network, then save the weights of the layers in neural network. On the next step I used that weights on a new LSTM neural network to train another LSTM neural network with one additional layer to fit better to the small dataset. Now we can better results using this technique.  <br/><br/>
The procedure is specially useful for languages which not have enough data for a proper modeling. In this project I used Farsi (Persion) comments from "digikala.com" translated to English by google translate. You can access dataset here: https://github.com/alisoltanirad/Sentiment-Analysis-Farsi-Dataset  <br/><br/>
* On the code you can change number of batch size and epochs of neural networks according to your computaion power on your system.  <br/><br/>