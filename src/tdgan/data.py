import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture,GaussianMixture
from sklearn.preprocessing import LabelEncoder
from tensorpack import DataFlow, RNGDataFlow
import logging
import json
from dateutil import parser
import functools
import time
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import logging
from tensorpack.utils import logger
from hmmlearn.hmm import GaussianHMM

logging.basicConfig(level = logging.INFO)
memoize=functools.lru_cache(maxsize=None)

class DateProcess:
    def __init__(self, date_columns, metadata):
        self.date_columns = date_columns
        self.metadata = metadata

    def split_date(self):
        date_parts = {}
        print("Inside Split function, the columns are")
        print(self.data.columns.tolist())
        for col in self.date_columns:
            hour, day, month, year = [], [], [], []
            for date_val in self.data[col].values:
                if not isinstance(date_val, str):
                    #print(f"Found datatype other than 'str' in {col} Date column, hence filling with nan values")
                    #print(date_val)
                    hour.append(np.NaN)
                    day.append(np.NaN)
                    month.append(np.NaN)
                    year.append(np.NaN)

                else:
                    date_val = parser.parse(date_val)
                    hour.append(date_val.hour)
                    day.append(date_val.day)
                    month.append(date_val.month)
                    year.append(date_val.year)
            date_parts.update({col+"_hour": hour, col+'_day': day, col+'_month':month, col+'_year': year})
        return date_parts

    def transform2cycles(self, meta):
        dataf = pd.DataFrame()
        metadata = {}
        date_gen=[]
        def tuple_(df):
            """Making series object or numpy.ndarray as mutable objects"""
            return tuple(df.to_dict().items())


        def inv_tuple(tuple_dict):
             return pd.Series([(y) for x,y in tuple_dict])

        def convert2cycles(arr, maxm):
            arr=inv_tuple(arr)
            sin_arr = [np.sin(2* np.pi * arr[i]/ maxm) for i in range(len(arr)) if i != np.NaN]
            cos_arr = [np.cos(2* np.pi * arr[i]/ maxm) for i in range(len(arr)) if i != np.NaN]
            return sin_arr, cos_arr


        for col_name in self.date_columns:
            dataf[col_name+"_hoursin"], dataf[col_name+'_hourcos'] = convert2cycles(tuple_(meta[col_name+'_hour']), 24)
            dataf[col_name+"_daysin"], dataf[col_name+"_daycos"] = convert2cycles(tuple_(meta[col_name+"_day"]), 31)
            dataf[col_name+"_monthsin"], dataf[col_name+"_monthcos"] = convert2cycles(tuple_(meta[col_name+"_month"]), 12)
            #dataf[col_name+"_yearsin"], dataf[col_name+"_yearcos"] = convert2cycles(tuple_(meta[col_name+"_year"]), max(meta[col_name+"_year"]))
            dataf["%s_year"%col_name]=meta["%s_year"%col_name]
            [date_gen.append(i) for i in dataf.columns if i != "%s_year"%col_name]
        logger.info(date_gen)
        metadata.update({"datecolumns":date_gen})
        #    metadata.update({col_name+"_year":max(meta[col_name+"_year"])})
        return dataf, metadata

    def interpolate(self, data):
        self.data = data
        date_meta = self.split_date()
        data_temp = self.data.drop(self.date_columns, axis= 1)

        def make_interpolation(data, col_name):
            try:
                data.sort_values(col_name)
            except:
                raise ValueError(f"The column name for study name {col_name} not found, check config file")
            sorted_val = data[col_name].unique()
            for value in sorted_val:
                data.loc[data[col_name] == value]=data.loc[data[col_name]==value].interpolate(limit_direction='both')
            return data
        
        for col_name in tqdm(date_meta):
            data_temp[f"{col_name}"] = date_meta[col_name]
        # if self.study_name:
        #     data_temp = make_interpolation(data_temp, self.study_name)
        for value in date_meta:
            date_meta[value] = data_temp[value]
        del data_temp
        data, metadata = self.transform2cycles(date_meta)
        date_meta['columns'] = data.columns
        self.metadata.update(metadata)
        self.data= pd.concat([self.data.drop(self.date_columns, axis = 1), data], axis = 1)
        return self.data, self.metadata



class Preprocessing:
    def __init__(self, data, config):
        self.continuous_cols = config['NUMERICAL_COLS']
        self.date_columns = config['DATE_COLS']
        self.date_delimiter = config['DATE_DELIMITER']
        self.metadata = {}
        self.data = data
        self.remove_cols = []
        for col in self.data.columns.tolist():
            val_len = len(self.data[col].value_counts())
            if val_len == 0:
                self.remove_cols.append(col)
                if col in self.continuous_cols:
                    del self.continuous_cols[self.continuous_cols.index(col)]
                if self.date_columns:
                    if col in self.date_columns:
                        del self.date_columns[self.date_columns.index(col)]
        self.data.drop(self.remove_cols, axis =1, inplace= True)
        self.columns = self.data.columns.tolist()
        if self.date_columns:
            config.update({"final_date_columns":self.date_columns})
        else:
            pass

        with open("caches/%s.json"%"config","w") as file:
            file.write(json.dumps(config))

        
        
        if self.date_columns:
            self.categorical_cols = list(set(self.columns) - set(self.continuous_cols + self.date_columns))
            self.date_processor = DateProcess(self.date_columns, self.metadata)
        else:
            self.categorical_cols = list(set(self.columns) - set(self.continuous_cols))
    def fill_nan(self, dataf, col_names, categorical = False):
        for col in col_names:
            if dataf[col].isnull().values.any():
                if categorical == True:
                    dataf[col] = dataf[col].astype(str)
                    replace_val = 'NULL'
                    dataf[col].fillna(replace_val, inplace = True)
        
                else:
                    try:
                        #val = dataf[col].astype(float).mean()
                        val = 0
                        dataf[col].fillna(val, inplace = True)
                    except Exception as E:
                        print(f"EXCEPTION: Can't take mean for the value in the {col} column as it raises the following {E}")
      
        return dataf

    def change_column_name(self):
        columns= self.data.columns.tolist()
        #print(columns)
        columns_names = {j:i for i,j in enumerate(columns)}
        self.metadata['columns']= columns_names
        self.data.rename(columns= {j: i for i, j in enumerate(columns)}, inplace = True)

    def get_data(self):
        self.data = self.fill_nan(self.data, self.continuous_cols)
        logger.info("nan filled")       
        if self.date_columns:
            for col in self.date_columns:
                self.categorical_cols.append(col+"_year") 
            self.data, self.metadata = self.date_processor.interpolate(self.data)
        self.data = self.fill_nan(self.data, self.categorical_cols, categorical = True)

        self.change_column_name()
        if self.continuous_cols:
            self.continuous_columns=[self.metadata["columns"][i] for i in self.continuous_cols]
        logger.info(self.continuous_columns)
        self.metadata["continous_columns"] = self.continuous_columns
        return self.data,self.metadata,self.continuous_columns



####

class DatePostProcess:
    def __init__(self, dataF, metafile, CONFIG):
        self.dataF = dataF
        self.config = CONFIG
        with open(metafile, 'r') as read_file:
            self.meta_data = json.load(read_file)
        
    def decode_date(self, sin_arr, cos_arr, maxm, part):
        decoded = []
        def convert_cycles(sin_val, cos_val, maxm):
            val1 = np.arcsin(sin_val)*maxm/(2*np.pi)
            val2 = np.arccos(cos_val)*maxm/(2*np.pi)
            return (val1, val2)
        
        def check_conditions(sin_arr, cos_arr):
            for i in range(len(sin_arr)):
                sin_arr[i] = float(sin_arr[i])
                cos_arr[i] = float(cos_arr[i])
                
                sin_arr[i] = 1 if sin_arr[i] >1 else sin_arr[i]
                sin_arr[i] = -1 if sin_arr[i] < -1 else sin_arr[i]
                
                cos_arr[i] = 1 if cos_arr[i] > 1 else cos_arr[i]
                cos_arr[i] = 1 if cos_arr[i] < -1 else cos_arr[i]
            return sin_arr, cos_arr

        sin_arr, cos_arr = check_conditions(sin_arr, cos_arr)
        for i in range(len(sin_arr)):
            sin_arr[i] = float(sin_arr[i])
            cos_arr[i] = float(cos_arr[i])
            
            if (np.isnan(sin_arr[i]) or np.isnan(cos_arr[i])) and (part == "day" or part == "month"):
                decoded.append(1)
            elif (round(sin_arr[i]) == 0 and round(cos_arr[i]) == 0) and  (part == "day" or part == "month"):
                decoded.append(1)
            elif (np.isnan(sin_arr[i]) or np.isnan(cos_arr[i])) and (part == "hour"):
                decoded.append(0) 
            else:

                sin_val, cos_val = convert_cycles(sin_arr[i], cos_arr[i], maxm)
                
                if sin_val >= 0 and cos_val>=0:
                    if (round(cos_val) == 0) and (part == "day" or part == "month"):
                        decoded.append(round(cos_val) + 1)
                    else:
                        decoded.append(round(cos_val))
                elif (sin_val>=0 and cos_val<=0) or (sin_val<=0 and cos_val>=0):
                    if (round(maxm - cos_val) == 0) and (part == "day" or part == "month"):
                        decoded.append(round(maxm - cos_val) + 1)
                    else:
                        decoded.append(round(maxm - cos_val))
                
                # else:
                #     decoded.append(round(maxm- cos_val))
                
        return np.asarray(decoded,dtype=int)


    def post_process_dates(self,date_exists=True):
        columns={i:j for i,j in enumerate(self.meta_data["columns"])}
        self.dataF.rename(columns=columns,inplace=True)

        if date_exists:
            date= ["hour","day","month","year"]
            self.temp_dataF = self.dataF
            data_  =self.inverse_transform()
            final_data = pd.DataFrame()
            for name in self.datecolumns:
                final_data[name] = data_[[name+'_day',name+'_month',name+'_year']].astype(str).agg('-'.join, axis=1)
            for name in self.datecolumns:
                for event in  date:
                    if event != "year":
                        for symbol in ["sin","cos"]:
                            self.temp_dataF.drop(name+"_"+event+symbol,axis=1,inplace=True)
                    else:
                        self.temp_dataF.drop(name+"_"+event,axis=1,inplace=True)
            final = pd.concat([self.temp_dataF,final_data],axis=1)
            return final
        else:
            return self.dataF
    #df_tdgan = post_process_dates(new_samples, "metafile.json", "Config.json")

    def inverse_transform(self):
        df=pd.DataFrame()
        self.datecolumns= self.config["DATE_COLS"]
        print(self.dataF.columns)
        for col in self.datecolumns:
            df[col+"_hour"]= self.decode_date(np.array(self.dataF[col+"_hoursin"]),np.array(self.dataF[col+"_hourcos"]),24, part="hour")
            df[col+"_day"]= self.decode_date(np.array(self.dataF[col+"_daysin"]),np.array(self.dataF[col+"_daycos"]),31, part="day")
            df[col+"_month"]= self.decode_date(np.array(self.dataF[col+"_monthsin"]),np.array(self.dataF[col+"_monthcos"]),12, part="month")
            #df[col+"_year"]=self.decode_date(np.array(self.dataF[col+"_yearsin"]),np.array(self.dataF[col+"_yearcos"]),self.meta_data[col+"_year"] , year=True)
            df[col+"_year"]=self.dataF[col+"_year"]
        return  df


def check_metadata(metadata):
    """Check that the given metadata has correct types for all its members.

    Args:
        metadata(dict): Description of the inputs.

    Returns:
        None

    Raises:
        AssertionError: If any of the details is not valid.

    """
    message = 'The given metadata contains unsupported types.'
    assert all([item['type'] in ['category', 'value'] for item in metadata['details']]), message


def check_inputs(function):
    """Validate inputs for functions whose first argument is a numpy.ndarray with shape (n,1).

    Args:
        function(callable): Method to validate.

    Returns:
        callable: Will check the inputs before calling :attr:`function`.

    Raises:
        ValueError: If first argument is not a valid :class:`numpy.array` of shape (n, 1).

    """
    def decorated(self, data, *args, **kwargs):
        if not (isinstance(data, np.ndarray) and len(data.shape) == 2 and data.shape[1] == 1):
            raise ValueError('The argument `data` must be a numpy.ndarray with shape (n, 1).')

        return function(self, data, *args, **kwargs)

    decorated.__doc__ = function.__doc__
    return decorated


def cluster_select(data):
    sil={}
    clusters=12
    only_one = False
    for n in tqdm(range(3,clusters)):
        gmm=GaussianMixture(n_components=n,max_iter=1000,tol=1e-4)
        try:
            op=gmm.fit_predict(data)
        except:
            #print("this is data",data)
            pass
        try:
            sil[n]=silhouette_score(data,op)
        except:
            only_one = True
            sil[n] = 1
    logger.info(sil)
    max_score_cluster= 0 if only_one == True else (np.argmax([sil[i] for i in range(3,len(sil))])+2)
    return max_score_cluster+1


class TGANDataFlow(RNGDataFlow):
    """Subclass of :class:`tensorpack.RNGDataFlow` prepared to work with :class:`numpy.ndarray`.

    Attributes:
        shuffle(bool): Wheter or not to shuffle the data.
        metadata(dict): Metadata for the given :attr:`data`.
        num_features(int): Number of features in given data.
        data(list): Prepared data from :attr:`filename`.
        distribution(list): DepecrationWarning?

    """

    def __init__(self, data, metadata, shuffle=True):
        """Initialize object.

        Args:
            filename(str): Path to the json file containing the metadata.
            shuffle(bool): Wheter or not to shuffle the data.

        Raises:
            ValueError: If any column_info['type'] is not supported

        """
        self.shuffle = shuffle
        if self.shuffle:
            self.reset_state()

        self.metadata = metadata
        self.num_features = self.metadata['num_features']

        self.data = []
        for col_id, col_info in enumerate(self.metadata['details']):
            if col_info['type'] == 'value':
                col_data = np.array(data['f%02d' % col_id])
                dimensions = col_data[:, :1]
                probability = col_data[:, 1:]
                self.data.append(dimensions)
                self.data.append(probability)

            elif col_info['type'] == 'category':
                col_data = np.asarray(data['f%02d' % col_id],dtype='int32')
                self.data.append(col_data)

            else:
                raise ValueError(
                    "column_info['type'] must be either 'category' or 'value'."
                    "Instead it was '{}'.".format(col_info['type'])
                )

        self.data = list(zip(*self.data))

    def size(self):
        """Return the number of rows in data.

        Returns:
            int: Number of rows in :attr:`data`.

        """
        return len(self.data)

    def get_data(self):
        """Yield the rows from :attr:`data`.

        Yields:
            tuple: Row of data.

        """
        idxs = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(idxs)

        for k in idxs:
            yield self.data[k]

    def __iter__(self):
        """Iterate over self.data."""
        return self.get_data()

    def __len__(self):
        """Length of batches."""
        return self.size()




class RandomZData(DataFlow):
    """Random dataflow.

    Args:
        shape(tuple): Shape of the array to return on :meth:`get_data`

    """

    def __init__(self, shape):
        """Initialize object."""
        super(RandomZData, self).__init__()
        self.shape = shape

    def get_data(self):
        """Yield random normal vectors of shape :attr:`shape`."""
        while True:
            yield [np.random.normal(0, 1, size=self.shape)]

    def __iter__(self):
        """Return data."""
        return self.get_data()

    def __len__(self):
        """Length of batches."""
        return self.shape[0]


class MultiModalNumberTransformer:
    r"""
    Args:
        num_modes(int): Number of modes on given data.
        tol:default to 1e -3 
        (if tol <(lower_bound-prev_lower_bound) then it is converged)
        max_iter:default to 100
    Attributes:
        num_modes(int): Number of components in the `skelarn.mixture.GaussianMixture`_ model.

    .. _skelarn.mixture.GaussianMixture: https://scikit-learn.org/stable/modules/generated/
        sklearn.mixture.GaussianMixture.html

    """

    def __init__(self, num_modes=5):
        """Initialize instance."""
        self.num_modes = num_modes

    @check_inputs
    def transform(self, data):
        """Cluster values using a `skelarn.mixture.GaussianMixture`_ model.

        Args:
            data(numpy.ndarray): Values to cluster in array of shape (n,1).

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, list, list]: Tuple containg the features,
            probabilities, averages and stds of the given data.

        .. _skelarn.mixture.BayesianGaussianMixture: https://scikit-learn.org/stable/modules/generated/
            sklearn.mixture.BayesianGaussianMixture.html

        """
        model =BayesianGaussianMixture(n_components=self.num_modes,max_iter=1000,tol=1e-4)
        model.fit(data)

        means = model.means_.reshape((1, self.num_modes))
        stds = np.sqrt(model.covariances_).reshape((1, self.num_modes))

        features = (data - means) / (2 * stds)
        probs = model.predict_proba(data)
        argmax = np.argmax(probs, axis=1)
        idx = np.arange(len(features))
        features = features[idx, argmax].reshape([-1, 1])

        features = np.clip(features, -0.99, 0.99)

        return features, probs, list(means.flat), list(stds.flat)

    def date_transform(self, data):
        """Cluster values using a `skelarn.mixture.GaussianMixture`_ model.

        Args:
            data(numpy.ndarray): Values to cluster in array of shape (n,1).

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, list, list]: Tuple containg the features,
            probabilities, averages and stds of the given data.

        .. _GaussianHMM: https://hmmlearn.readthedocs.io/en/latest/api.html#hmmlearn.hmm.GaussianHMM

        """
        model =GaussianHMM(n_components=4)
        model.fit(data)

        means = model.means_.reshape((1, self.num_modes))
        stds = np.sqrt(model.covars_).reshape((1, self.num_modes))

        features = (data - means) / (2 * stds)
        probs = model.predict_proba(data)
        argmax = np.argmax(probs, axis=1)
        idx = np.arange(len(features))
        features = features[idx, argmax].reshape([-1, 1])

        features = np.clip(features, -0.99, 0.99)

        return features, probs, list(means.flat), list(stds.flat)

    @staticmethod
    def inverse_transform(data, info):
        """Reverse the clustering of values.

        Args:
            data(numpy.ndarray): Transformed data to restore.
            info(dict): Metadata.

        Returns:
           numpy.ndarray: Values in the original space.

        """
        features = data[:, 0]
        probs = data[:, 1:]
        p_argmax = np.argmax(probs, axis=1)

        mean = np.asarray(info['means'])
        std = np.asarray(info['stds'])

        select_mean = mean[p_argmax]
        select_std = std[p_argmax]

        return features * 2 * select_std + select_mean


class Preprocessor:
    """Transform back and forth human-readable data into TGAN numerical features.

    Args:
        continous_columns(list): List of columns to be considered continuous
        metadata(dict): Metadata to initialize the object.

    Attributes:
        continous_columns(list): Same as constructor argument.
        metadata(dict): Information about the transformations applied to the data and its format.
        continous_transformer(MultiModalNumberTransformer):
            Transformer for columns in :attr:`continuous_columns`
        categorical_transformer(CategoricalTransformer):
            Transformer for categorical columns.
        columns(list): List of columns labels.

    """

    def __init__(self, continuous_columns=None, metadata=None,datecolumns=None):
        """Initialize object, set arguments as attributes, initialize transformers."""
        if continuous_columns is None:
            continuous_columns = []
        
        self.datecolumns=datecolumns
        if self.datecolumns is None:
             self.datecolumns=[]
        self.continuous_columns = continuous_columns
        self.metadata = metadata
        self.categorical_transformer = LabelEncoder()
        self.continous_transformer = MultiModalNumberTransformer()
        self.columns = None

    def fit_transform(self, data, fitting=True,get_data=False):
        """Transform human-readable data into TGAN numerical features.

        Args:
            data(pandas.DataFrame): Data to transform.
            fitting(bool): Whether or not to update self.metadata.

        Returns:
            pandas.DataFrame: Model features

        """
        num_cols = data.shape[1]
        #logger.info("Hi",type(num_cols))

        self.columns = data.columns
        data.columns = list(range(num_cols))
        
        transformed_data = {}
        details = []
        
        logger.info("Generating Clusters for Continous columns-{}".format(self.continuous_columns))
        for i in data.columns:
            if i in self.continuous_columns:
                column_data = data[i].values.reshape([-1, 1])
                logger.info("Generating Cluster for Continous column-{}".format(i))
#                try:
                # modes=cluster_select(column_data)
                # # #modes=7
                # # logger.info(modes)
                # self.continous_transformer = MultiModalNumberTransformer(modes)
#                except:
                modes = 4
                self.continous_transformer=MultiModalNumberTransformer(modes)

                features, probs, means, stds = self.continous_transformer.transform(column_data)
                transformed_data['f%02d' % i] = np.concatenate((features, probs), axis=1)

                if fitting:
                    details.append({
                        "type": "value",
                        "means": means,
                        "stds": stds,
                        "n": int(modes)
                    })
            elif i in self.datecolumns:
                column_data = data[i].values.reshape([-1, 1])
                logger.info("Generating Cluster for date column-{}".format(i))
#                try:
                #modes=cluster_select(column_data)
                #logger.info(modes)
                self.continous_transformer = MultiModalNumberTransformer(4)
#                except:
#                    modes=2
#                    self.continous_transformer=MultiModalNumberTransformer(modes)

                features, probs, means, stds = self.continous_transformer.date_transform(column_data)
                transformed_data['f%02d' % i] = np.concatenate((features, probs), axis=1)

                if fitting:
                    details.append({
                        "type": "value",
                        "means": means,
                        "stds": stds,
                        "n": 4
                    })
            else:
                logger.info("Generating clusters for Discrete column-{}".format(i))
                column_data = data[i].astype(str).values
                
                features = self.categorical_transformer.fit_transform(column_data)
                
                transformed_data['f%02d' % i] = features.reshape([-1, 1])

                if fitting:
                    mapping = self.categorical_transformer.classes_
                    #print(mapping)
                    details.append({
                        "type": "category",
                        "mapping": mapping,
                        "n": int(mapping.shape[0]),
                    })

        if fitting:
            metadata = {
                "num_features": num_cols,
                "details": details
            }
            check_metadata(metadata)
            self.metadata = metadata
            
        return transformed_data

    def transform(self, data):
        """Transform the given dataframe without generating new metadata.

        Args:
            data(pandas.DataFrame): Data to fit the object.

        """
        return self.fit_transform(data, fitting=False)

    def fit(self, data):
        """Initialize the internal state of the object using :attr:`data`.

        Args:
            data(pandas.DataFrame): Data to fit the object.

        """
        self.fit_transform(data)

    def reverse_transform(self, data):
        """Transform TGAN numerical features back into human-readable data.

        Args:
            data(pandas.DataFrame): Data to transform.
            fitting(bool): Whether or not to update self.metadata.

        Returns:
            pandas.DataFrame: Model features

        """
        table = []
        
        for i in range(self.metadata['num_features']):
            column_data = np.array(data['f%02d' % i])
            column_metadata = self.metadata['details'][i]
            
            if column_metadata['type'] == 'value':
                column = self.continous_transformer.inverse_transform(column_data, column_metadata)
                # column = list(map(lambda x: 0 if x<0 else round(x) , column))

            if column_metadata['type'] == 'category':
                self.categorical_transformer.classes_ = np.asarray(column_metadata["mapping"])
                
                column_data=np.asarray(column_data,dtype=np.int32)
                
                column = self.categorical_transformer.inverse_transform(column_data.ravel().astype(np.int32))

            table.append(column)

        result = pd.DataFrame(dict(enumerate(table)))
        return result
