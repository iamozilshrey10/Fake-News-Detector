{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Import Modules\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import itertools \n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import PassiveAggressiveClassifier\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>8476</td>\n",
       "      <td>You Can Smell Hillary’s Fear</td>\n",
       "      <td>Daniel Greenfield, a Shillman Journalism Fello...</td>\n",
       "      <td>FAKE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>10294</td>\n",
       "      <td>Watch The Exact Moment Paul Ryan Committed Pol...</td>\n",
       "      <td>Google Pinterest Digg Linkedin Reddit Stumbleu...</td>\n",
       "      <td>FAKE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>3608</td>\n",
       "      <td>Kerry to go to Paris in gesture of sympathy</td>\n",
       "      <td>U.S. Secretary of State John F. Kerry said Mon...</td>\n",
       "      <td>REAL</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>10142</td>\n",
       "      <td>Bernie supporters on Twitter erupt in anger ag...</td>\n",
       "      <td>— Kaydee King (@KaydeeKing) November 9, 2016 T...</td>\n",
       "      <td>FAKE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>875</td>\n",
       "      <td>The Battle of New York: Why This Primary Matters</td>\n",
       "      <td>It's primary day in New York and front-runners...</td>\n",
       "      <td>REAL</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0                                              title  \\\n",
       "0        8476                       You Can Smell Hillary’s Fear   \n",
       "1       10294  Watch The Exact Moment Paul Ryan Committed Pol...   \n",
       "2        3608        Kerry to go to Paris in gesture of sympathy   \n",
       "3       10142  Bernie supporters on Twitter erupt in anger ag...   \n",
       "4         875   The Battle of New York: Why This Primary Matters   \n",
       "\n",
       "                                                text label  \n",
       "0  Daniel Greenfield, a Shillman Journalism Fello...  FAKE  \n",
       "1  Google Pinterest Digg Linkedin Reddit Stumbleu...  FAKE  \n",
       "2  U.S. Secretary of State John F. Kerry said Mon...  REAL  \n",
       "3  — Kaydee King (@KaydeeKing) November 9, 2016 T...  FAKE  \n",
       "4  It's primary day in New York and front-runners...  REAL  "
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Read data and get shape for first 5 records\n",
    "\n",
    "data = pd.read_csv('C:\\\\Users\\\\sarta\\\\Desktop\\\\Fake_News\\\\news.csv')\n",
    "data.shape\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    FAKE\n",
       "1    FAKE\n",
       "2    REAL\n",
       "3    FAKE\n",
       "4    REAL\n",
       "Name: label, dtype: object"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Get only labels\n",
    "\n",
    "labels = data.label\n",
    "labels.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Split the dataset into training and test sets\n",
    "\n",
    "X_train,X_test,y_train,y_test=train_test_split(data['text'],\n",
    "                                              labels,\n",
    "                                              test_size=0.2,\n",
    "                                              random_state=7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Initializing a TfidVectorizer"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
