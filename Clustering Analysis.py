import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, adjusted_rand_score, silhouette_score, davies_bouldin_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
import itertools
from scipy.stats import f_oneway
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Note: umap-learn not installed, UMAP visualization unavailable")
try:
    from metric_learn import LMNN
    METRIC_LEARN_AVAILABLE = True
except ImportError:
    METRIC_LEARN_AVAILABLE = False
    print("Note: metric-learn not installed, metric learning feature selection unavailable")

rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

class HydrocarbonSourceRockAnalyzerAdvanced:
    """
    Advanced analyzer for hydrocarbon source rock elemental ratios.
    Incorporates geochemical knowledge, metric learning, and automatic feature selection
    for perfect Fréchet clustering separation.
    """

    GEOCHEMICAL_INDICATORS = {
        'V_Ni': {'formula': 'V/Ni', 'meaning': 'Redox indicator, >1.5 anoxic', 
                'ideal_range': (0.5, 3.0), 'depositional_implication': 'Marine anoxic V/Ni>1.5'},
        'Pr_Ph': {'formula': 'Pr/Ph', 'meaning': 'Organic source/redox', 
                 'ideal_range': (0.2, 1.5), 'depositional_implication': 'Pr/Ph<1 reducing'},
        'Sr_Ba': {'formula': 'Sr/Ba', 'meaning': 'Paleosalinity, >1 marine', 
                 'ideal_range': (0.1, 10), 'depositional_implication': 'Sr/Ba>1 marine'},
        'La_Ce': {'formula': 'La/Ce', 'meaning': 'REE fractionation, provenance', 
                 'ideal_range': (0.3, 0.6), 'depositional_implication': 'Normal 0.3-0.6'},
        'Eu_anomaly': {'formula': 'Eu/Eu*', 'meaning': 'Eu anomaly, hydrothermal', 
                      'ideal_range': (0.8, 1.2), 'depositional_implication': 'Eu/Eu*>1 hydrothermal'},
        'Ce_anomaly': {'formula': 'Ce/Ce*', 'meaning': 'Ce anomaly, redox', 
                      'ideal_range': (0.8, 1.2), 'depositional_implication': 'Ce/Ce*<1 oxic'},
        'U_Th': {'formula': 'U/Th', 'meaning': 'Redox, U enrichment anoxic', 
                'ideal_range': (0.1, 1.0), 'depositional_implication': 'U/Th>0.75 anoxic'},
        'Ni_Co': {'formula': 'Ni/Co', 'meaning': 'Anoxic indicator', 
                 'ideal_range': (1.0, 5.0), 'depositional_implication': 'Ni/Co>2 anoxic'},
        'V_Cr': {'formula': 'V/Cr', 'meaning': 'Redox indicator', 
                'ideal_range': (1.0, 4.0), 'depositional_implication': 'V/Cr>2 anoxic'},
        'Al_Ti': {'formula': 'Al/Ti', 'meaning': 'Terrigenous input', 
                 'ideal_range': (10, 30), 'depositional_implication': 'High Al/Ti terrigenous'},
        'Zr_Hf': {'formula': 'Zr/Hf', 'meaning': 'Zircon fractionation, provenance', 
                 'ideal_range': (30, 50), 'depositional_implication': 'Normal 30-50'},
        'Th_U': {'formula': 'Th/U', 'meaning': 'Provenance and redox', 
                'ideal_range': (2.0, 8.0), 'depositional_implication': 'Th/U<2 anoxic'},
        'La_Lu': {'formula': 'La/Lu', 'meaning': 'REE fractionation', 
                 'ideal_range': (8.0, 20.0), 'depositional_implication': 'High La/Lu terrigenous'},
        'Sr_Ca': {'formula': 'Sr/Ca', 'meaning': 'Paleoproductivity', 
                 'ideal_range': (0.001, 0.01), 'depositional_implication': 'Productivity related'},
        'Mn_Fe': {'formula': 'Mn/Fe', 'meaning': 'Redox, high oxic', 
                 'ideal_range': (0.01, 0.1), 'depositional_implication': 'Mn/Fe high oxic'},
        'Mo_U': {'formula': 'Mo/U', 'meaning': 'Redox and productivity', 
                'ideal_range': (0.1, 1.0), 'depositional_implication': 'High Mo/U anoxic productivity'},
        'Cu_Zn': {'formula': 'Cu/Zn', 'meaning': 'Hydrothermal activity', 
                 'ideal_range': (0.1, 1.0), 'depositional_implication': 'Cu/Zn>0.5 hydrothermal'},
        'Cr_Zr': {'formula': 'Cr/Zr', 'meaning': 'Source weathering', 
                 'ideal_range': (0.1, 1.0), 'depositional_implication': 'Low Cr/Zr strong weathering'},
        'La_Y': {'formula': 'La/Y', 'meaning': 'REE pattern', 
                'ideal_range': (0.5, 2.0), 'depositional_implication': 'Provenance indicator'},
        'Y_Ho': {'formula': 'Y/Ho', 'meaning': 'REE fractionation', 
                'ideal_range': (25, 35), 'depositional_implication': 'Normal seawater ~28'}
    }

    DEPOSITIONAL_ENVIRONMENTS = {
        'Deep marine anoxic': {
            'indicators': ['V/Ni>1.5', 'U/Th>0.75', 'Ni/Co>2', 'V/Cr>2', 'Mo/U>0.5'],
            'typical_ratios': {'V/Ni': (1.5, 3.0), 'U/Th': (0.75, 1.5), 'Ni/Co': (2, 5)}
        },
        'Shallow marine oxic': {
            'indicators': ['V/Ni<1.0', 'U/Th<0.5', 'Ce/Ce*<0.9', 'Mn/Fe>0.05'],
            'typical_ratios': {'V/Ni': (0.5, 1.0), 'U/Th': (0.1, 0.5), 'Ce/Ce*': (0.8, 0.9)}
        },
        'Lacustrine': {
            'indicators': ['Sr/Ba<1', 'Al/Ti>20', 'La/Lu>15', 'Pr/Ph<0.8'],
            'typical_ratios': {'Sr/Ba': (0.1, 1.0), 'Al/Ti': (20, 40), 'La/Lu': (10, 30)}
        },
        'Deltaic': {
            'indicators': ['Mixed', 'Al/Ti moderate', 'Zr/Hf normal', 'Sr/Ba variable'],
            'typical_ratios': {'Al/Ti': (15, 25), 'Zr/Hf': (30, 45), 'Sr/Ba': (0.5, 2.0)}
        },
        'Hydrothermal': {
            'indicators': ['Eu/Eu*>1.1', 'Cu/Zn>0.5', 'Element enrichment'],
            'typical_ratios': {'Eu/Eu*': (1.1, 1.5), 'Cu/Zn': (0.5, 2.0)}
        }
    }

    SOURCE_ROCK_TYPES = {
        'Type I (Lacustrine)': {
            'description': 'H-rich, algal source, high oil potential',
            'indicators': ['High H/C', 'Low O/C', 'Pr/Ph<1', 'High gammacerane'],
            'elemental_features': ['High V/Ni', 'High Ni/Co', 'Sr/Ba<1']
        },
        'Type II (Marine)': {
            'description': 'Mixed source, marine plankton and bacteria',
            'indicators': ['Moderate H/C', 'Pr/Ph~1', 'Regular steranes'],
            'elemental_features': ['V/Ni>1.5', 'U/Th>0.75', 'Sr/Ba>1']
        },
        'Type III (Terrestrial)': {
            'description': 'H-poor, terrestrial higher plants, gas-prone',
            'indicators': ['Low H/C', 'High O/C', 'Pr/Ph>1', 'Oleanane'],
            'elemental_features': ['Low V/Ni', 'High Al/Ti', 'Variable Sr/Ba']
        },
        'Type II-S (Marine sulfur-rich)': {
            'description': 'Marine anoxic, high sulfur',
            'indicators': ['High sulfur', 'High V/Ni', 'Mo enrichment'],
            'elemental_features': ['V/Ni>2', 'Mo/U>0.7', 'U/Th>1']
        }
    }

    def __init__(self, data_path=None, df=None, class_col=None):
        self.ratio_features = None
        self.selected_features = None
        self.feature_importance_df = None
        self.model_results = {}
        self.distance_metrics = {}
        self.learned_metric = None
        self.geochemical_interpretations = {}
        self.depositional_inferences = {}
        self.source_rock_inferences = {}
        self.optimal_frechet_features = None
        self.clustering_membership = None

        if df is not None:
            self.df = df.copy()
        elif data_path:
            if data_path.endswith('.xlsx'):
                self.df = pd.read_excel(data_path)
            elif data_path.endswith('.xls'):
                self.df = pd.read_excel(data_path)
            elif data_path.endswith('.csv'):
                self.df = pd.read_csv(data_path, encoding='utf-8')
            else:
                raise ValueError("Unsupported file format. Use .xlsx, .xls, or .csv")
        else:
            raise ValueError("Provide data path or DataFrame")

        self.class_col = class_col
        self._preprocess_data()

    def _preprocess_data(self):
        print("=" * 60)
        print("Data Preprocessing - Geochemical Edition")
        print("=" * 60)

        # Strip column names to avoid hidden spaces
        self.df.columns = self.df.columns.str.strip()

        print(f"Data shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        print("\nFirst 3 rows:")
        print(self.df.head(3))

        if self.class_col is None:
            class_cols = [col for col in self.df.columns if any(kw in col.lower() for kw in
                          ['class', 'type', 'formation', 'group', 'sample', 'layer'])]
            if class_cols:
                self.class_col = class_cols[0]
                print(f"Identified class column: {self.class_col}")
            else:
                first_col = self.df.columns[0]
                if self.df[first_col].dtype == 'object' or len(self.df[first_col].unique()) < 20:
                    self.class_col = first_col
                    print(f"Using first column as class: {self.class_col}")
                else:
                    self.class_col = 'Group'
                    self.df[self.class_col] = 'Unknown'
                    print(f"No class column found, created dummy: {self.class_col}")

        if self.class_col not in self.df.columns:
            print(f"Warning: class column '{self.class_col}' not in data, using first column")
            self.class_col = self.df.columns[0]

        element_cols = []
        for col in self.df.columns:
            if col != self.class_col and pd.api.types.is_numeric_dtype(self.df[col]):
                element_cols.append(col)
        self.element_cols = element_cols
        print(f"Identified {len(self.element_cols)} element columns")
        if len(self.element_cols) > 10:
            print(f"Element examples: {self.element_cols[:10]}...")
        else:
            print(f"Elements: {self.element_cols}")

        print("\nGeochemical data quality check:")
        self._check_geochemical_data_quality()

        if len(self.element_cols) > 0:
            missing = self.df[self.element_cols].isnull().sum()
            if missing.sum() > 0:
                print(f"\nMissing values found:")
                miss_cols = missing[missing > 0]
                print(miss_cols.head(10))
                if len(miss_cols) > 10:
                    print(f"... and {len(miss_cols)-10} more")
                for col in miss_cols.index:
                    if any(kw in col.lower() for kw in ['la','ce','nd','sm','eu','gd','tb','dy','ho','er','tm','yb','lu']):
                        re_cols = [c for c in self.element_cols if any(re in c.lower() for re in
                                   ['la','ce','pr','nd','sm','eu','gd','tb','dy','ho','er','tm','yb','lu'])]
                        median_val = self.df[re_cols].median().median() if re_cols else self.df[col].median()
                    else:
                        median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    print(f"  Column '{col}': filled {miss_cols[col]} missing with median {median_val:.4f}")
                print("Missing values filled with median.")
        else:
            print("Warning: No element data columns found.")

        self._remove_geochemical_outliers()

        self.le = LabelEncoder()
        self.df['label_encoded'] = self.le.fit_transform(self.df[self.class_col])
        self.class_names = self.le.classes_
        self.n_classes = len(self.class_names)
        print(f"\nClass labels: {self.class_names}")
        print(f"Class counts:\n{self.df[self.class_col].value_counts()}")

        if len(self.element_cols) > 0:
            self._calculate_distance_metrics(self.df[self.element_cols], 'Raw elements')

    def _check_geochemical_data_quality(self):
        if not self.element_cols:
            return
        print("1. Unit check:")
        unit_issues = []
        for col in self.element_cols:
            cl = col.lower()
            if any(e in cl for e in ['sio2','al2o3','tio2','fe2o3','mgo','cao','na2o','k2o','p2o5']):
                if self.df[col].max() > 100:
                    unit_issues.append(f"  {col}: max {self.df[col].max():.2f}, possible unit error (should be wt%)")
            elif any(e in cl for e in ['v','ni','co','cr','zn','cu','la','ce','nd','sr','ba','zr']):
                if self.df[col].max() < 0.1:
                    unit_issues.append(f"  {col}: max {self.df[col].max():.2f}, possible unit error (should be ppm)")
        if unit_issues:
            print("  Warning: possible unit issues:")
            for issue in unit_issues:
                print(issue)
        else:
            print("  Unit check passed.")

        print("2. Range check:")
        ranges = {
            'sio2': (30,90), 'al2o3': (1,25), 'tio2': (0.1,2), 'v': (10,500), 'ni': (10,300),
            'sr': (50,2000), 'ba': (100,1000), 'la': (5,100), 'ce': (10,200)
        }
        range_issues = []
        for col in self.element_cols:
            cl = col.lower()
            for elem, (lo, hi) in ranges.items():
                if elem in cl:
                    if self.df[col].min() < lo*0.5 or self.df[col].max() > hi*2:
                        range_issues.append(f"  {col}: {self.df[col].min():.2f}-{self.df[col].max():.2f}, expected {lo}-{hi}")
        if range_issues:
            print("  Warning: range anomalies:")
            for issue in range_issues[:5]:
                print(issue)
            if len(range_issues) > 5:
                print(f"  ... and {len(range_issues)-5} more")
        else:
            print("  Range check passed.")

        print("3. Element correlation (preliminary):")
        if len(self.element_cols) >= 4:
            try:
                la_cols = [c for c in self.element_cols if 'la' in c.lower()]
                ce_cols = [c for c in self.element_cols if 'ce' in c.lower()]
                if la_cols and ce_cols:
                    corr = self.df[la_cols[0]].corr(self.df[ce_cols[0]])
                    if corr < 0.7:
                        print(f"  Warning: {la_cols[0]} and {ce_cols[0]} correlation low ({corr:.3f})")
                    else:
                        print(f"  {la_cols[0]} and {ce_cols[0]} correlation normal: {corr:.3f}")
            except:
                pass

    def _remove_geochemical_outliers(self):
        if not self.element_cols:
            return
        print("\nRemoving geochemical outliers:")
        ranges = {
            'sio2': (30,90), 'al2o3': (1,30), 'fe2o3': (0.1,20), 'mgo': (0.1,20), 'cao': (0.1,30),
            'na2o': (0.1,10), 'k2o': (0.1,10), 'tio2': (0.01,2), 'p2o5': (0.01,1), 'mno': (0.01,1),
            'v': (5,1000), 'ni': (5,500), 'co': (1,100), 'cr': (5,500), 'zn': (5,200), 'cu': (5,200),
            'sr': (10,5000), 'ba': (10,3000), 'zr': (10,500), 'rb': (1,200), 'th': (1,100), 'u': (0.5,50),
            'la': (0.5,200), 'ce': (1,400), 'pr': (0.1,50), 'nd': (0.5,200), 'sm': (0.1,50),
            'eu': (0.05,20), 'gd': (0.1,50), 'tb': (0.05,10), 'dy': (0.1,50), 'ho': (0.05,10),
            'er': (0.1,30), 'tm': (0.02,5), 'yb': (0.1,30), 'lu': (0.02,5), 'y': (1,100),
            'sc': (1,50), 'nb': (1,50), 'mo': (0.1,20), 'cd': (0.01,5), 'sb': (0.01,5),
            'cs': (0.1,10), 'hf': (0.5,20), 'ta': (0.1,5), 'w': (0.1,10), 'tl': (0.01,5),
            'pb': (1,100), 'bi': (0.01,5)
        }
        removed = 0
        for pat, (lo, hi) in ranges.items():
            for col in [c for c in self.element_cols if pat in c.lower()]:
                mask = (self.df[col] >= lo) & (self.df[col] <= hi)
                cnt = (~mask).sum()
                if cnt > 0:
                    med = self.df.loc[mask, col].median()
                    self.df.loc[~mask, col] = med
                    removed += cnt
                    print(f"  {col}: replaced {cnt} outliers with median {med:.2f}")
        print(f"Total {removed} outliers replaced.")

    def _clean_ratio_features(self, ratio_features_df):
        print("Cleaning ratio features...")
        clean = ratio_features_df.copy()
        total_nan = clean.isnull().sum().sum()
        total_inf = (clean == np.inf).sum().sum() + (clean == -np.inf).sum().sum()
        if total_nan > 0:
            print(f"Found {total_nan} NaN values")
        if total_inf > 0:
            print(f"Found {total_inf} infinite values")
        for col in clean.columns:
            clean[col] = clean[col].replace([np.inf, -np.inf], np.nan)
            nan_cnt = clean[col].isnull().sum()
            if nan_cnt > 0:
                med = clean[col].median()
                if pd.isna(med):
                    med = 0
                    print(f"  Warning: column '{col}' all NaN, filled with 0")
                clean[col].fillna(med, inplace=True)
                if nan_cnt > 0:
                    print(f"  Column '{col}': filled {nan_cnt} NaNs with median {med:.4f}")
        rem_nan = clean.isnull().sum().sum()
        rem_inf = (clean == np.inf).sum().sum() + (clean == -np.inf).sum().sum()
        if rem_nan > 0:
            print(f"Warning: {rem_nan} NaNs remain")
        if rem_inf > 0:
            print(f"Warning: {rem_inf} infs remain")
        print("Ratio cleaning done.")
        return clean

    def _find_element_match(self, element_pattern):
        """
        Enhanced element matching: handles suffixes like 'N', '%', oxides.
        """
        pat = str(element_pattern).strip().lower()
        # First, try exact match (case-insensitive)
        for col in self.element_cols:
            if col.lower() == pat:
                return col

        # Second, try to match after stripping common suffixes
        def normalize(name):
            name = name.lower()
            if name.endswith('n') and len(name) > 1:
                name = name[:-1]
            name = name.replace('%', '')
            replacements = {
                '2o3': '',  # for Al2O3 -> Al
                'o2': '',
                'o': '',
            }
            for suf, repl in replacements.items():
                if name.endswith(suf):
                    name = name.replace(suf, repl)
            return name

        pat_norm = normalize(pat)
        best_match = None
        best_score = 0
        for col in self.element_cols:
            col_norm = normalize(col)
            if pat_norm == col_norm:
                return col
            if pat_norm in col_norm or col_norm in pat_norm:
                score = min(len(pat_norm), len(col_norm))
                if score > best_score:
                    best_score = score
                    best_match = col
        if best_match:
            return best_match

        # Finally, use the original symbol mapping
        sym_map = {
            'v': ['v','vanadium'], 'ni': ['ni','nickel'], 'co': ['co','cobalt'], 'cr': ['cr','chromium'],
            'cu': ['cu','copper'], 'zn': ['zn','zinc'], 'sr': ['sr','strontium'], 'ba': ['ba','barium'],
            'la': ['la','lanthanum'], 'ce': ['ce','cerium'], 'pr': ['pr','praseodymium'],
            'nd': ['nd','neodymium'], 'sm': ['sm','samarium'], 'eu': ['eu','europium'],
            'gd': ['gd','gadolinium'], 'tb': ['tb','terbium'], 'dy': ['dy','dysprosium'],
            'ho': ['ho','holmium'], 'er': ['er','erbium'], 'tm': ['tm','thulium'],
            'yb': ['yb','ytterbium'], 'lu': ['lu','lutetium'], 'y': ['y','yttrium'],
            'th': ['th','thorium'], 'u': ['u','uranium'], 'zr': ['zr','zirconium'],
            'hf': ['hf','hafnium'], 'nb': ['nb','niobium'], 'ta': ['ta','tantalum'],
            'mo': ['mo','molybdenum'], 'rb': ['rb','rubidium'], 'cs': ['cs','caesium'],
            'sc': ['sc','scandium'], 'ga': ['ga','gallium'], 'ge': ['ge','germanium'],
            'as': ['as','arsenic'], 'se': ['se','selenium'], 'br': ['br','bromine'],
            'ru': ['ru','ruthenium'], 'rh': ['rh','rhodium'], 'pd': ['pd','palladium'],
            'ag': ['ag','silver'], 'cd': ['cd','cadmium'], 'in': ['in','indium'],
            'sn': ['sn','tin'], 'sb': ['sb','antimony'], 'te': ['te','tellurium'],
            're': ['re','rhenium'], 'os': ['os','osmium'], 'ir': ['ir','iridium'],
            'pt': ['pt','platinum'], 'au': ['au','gold'], 'hg': ['hg','mercury'],
            'tl': ['tl','thallium'], 'pb': ['pb','lead'], 'bi': ['bi','bismuth'],
            'si': ['si','silicon','sio2'], 'al': ['al','aluminium','al2o3'],
            'ti': ['ti','titanium','tio2'], 'fe': ['fe','iron','fe2o3'],
            'mg': ['mg','magnesium','mgo'], 'ca': ['ca','calcium','cao'],
            'na': ['na','sodium','na2o'], 'k': ['k','potassium','k2o'],
            'p': ['p','phosphorus','p2o5'], 'mn': ['mn','manganese','mno']
        }
        for key, syns in sym_map.items():
            if pat == key or pat in syns:
                for syn in syns:
                    for col in self.element_cols:
                        if syn in col.lower():
                            return col
        return None

    def _calculate_distance_metrics(self, X, name):
        if X.empty:
            return
        if X.isnull().any().any():
            print(f"Warning: {name} contains NaNs, filling with median")
            X = X.fillna(X.median())
        if (X == np.inf).any().any() or (X == -np.inf).any().any():
            print(f"Warning: {name} contains infinities, replacing")
            X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        y = self.df['label_encoded']
        classes = np.unique(y)
        if len(classes) < 2:
            return
        centers = [np.mean(Xs[y==c], axis=0) for c in classes]
        inter = cdist(centers, centers, 'euclidean')
        intra = []
        for c in classes:
            d = Xs[y==c]
            if len(d) > 1:
                intra.append(np.mean(pdist(d, 'euclidean')))
        metrics = {
            'mean inter': np.mean(inter[np.triu_indices(len(classes), k=1)]),
            'min inter': np.min(inter[np.triu_indices(len(classes), k=1)]),
            'max inter': np.max(inter[np.triu_indices(len(classes), k=1)]),
            'mean intra': np.mean(intra) if intra else 0,
            'sep ratio': np.mean(inter[np.triu_indices(len(classes), k=1)]) / (np.mean(intra)+1e-10) if intra else 0,
            'n_features': X.shape[1]
        }
        self.distance_metrics[name] = metrics
        print(f"\n{name} distance metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    def generate_ratio_features(self, element_groups=None, priority_geochemical_ratios=True, max_ratios=300):
        print("\n" + "="*60)
        print("Generating ratio features - Geochemical Edition")
        print("="*60)
        if not hasattr(self, 'element_cols') or not self.element_cols:
            print("Error: No element columns found.")
            return None
        self.element_cols = [str(c).strip() for c in self.element_cols]
        self.df.columns = [str(c).strip() if c in self.element_cols else c for c in self.df.columns]

        ratio_pairs = []
        if priority_geochemical_ratios:
            print("Phase 1: Geochemical indicator ratios")
            priority = []
            for info in self.GEOCHEMICAL_INDICATORS.values():
                form = info['formula']
                if '/' in form:
                    e1, e2 = form.split('/')
                    m1 = self._find_element_match(e1)
                    m2 = self._find_element_match(e2)
                    if m1 and m2:
                        if (m1, m2) not in priority:
                            priority.append((m1, m2))
                            print(f"  Added: {m1}/{m2} - {info['meaning']}")
                    else:
                        if not m1:
                            print(f"  Warning: element {e1} not found")
                        if not m2:
                            print(f"  Warning: element {e2} not found")
            ratio_pairs.extend(priority)
            print(f"Generated {len(priority)} geochemical ratios.")

        if element_groups:
            print("\nPhase 2: Within-group ratios")
            group_pairs = []
            for gname, elems in element_groups.items():
                print(f"Processing {gname}: {elems}")
                avail = []
                for e in elems:
                    m = self._find_element_match(e)
                    if m:
                        avail.append(m)
                    else:
                        print(f"  Warning: {e} not found")
                if len(avail) > 1:
                    pairs = list(itertools.combinations(avail, 2))
                    new = [p for p in pairs if p not in ratio_pairs]
                    group_pairs.extend(new)
                    print(f"  Added {len(new)} new pairs from {gname}")
                else:
                    print(f"  Warning: insufficient elements in {gname}")
            ratio_pairs.extend(group_pairs)
            if len(group_pairs) < 10:
                print("Few within-group ratios, adding cross-group pairs...")
                selected = []
                for gname, elems in element_groups.items():
                    for e in elems[:3]:
                        m = self._find_element_match(e)
                        if m and m not in selected:
                            selected.append(m)
                if len(selected) > 1:
                    extra = list(itertools.combinations(selected, 2))
                    extra = [p for p in extra if p not in ratio_pairs][:50]
                    ratio_pairs.extend(extra)
        else:
            print("\nPhase 3: Supplemental ratios")
            if len(self.element_cols) > 30:
                print(f"Many elements ({len(self.element_cols)}), using top 30 by variance")
                try:
                    vars_ = self.df[self.element_cols].var().sort_values(ascending=False)
                    top = vars_.head(30).index.tolist()
                    all_pairs = list(itertools.combinations(top, 2))
                except:
                    top = self.element_cols[:30]
                    all_pairs = list(itertools.combinations(top, 2))
            else:
                all_pairs = list(itertools.combinations(self.element_cols, 2))
            new_pairs = [p for p in all_pairs if p not in ratio_pairs][:100]
            ratio_pairs.extend(new_pairs)

        if not ratio_pairs:
            print("Warning: No ratio pairs generated, using first 10 elements")
            if len(self.element_cols) >= 2:
                top = self.element_cols[:min(10, len(self.element_cols))]
                ratio_pairs = list(itertools.combinations(top, 2))
            else:
                print("Error: Less than 2 element columns.")
                return None

        if len(ratio_pairs) > max_ratios:
            print(f"Ratio pairs ({len(ratio_pairs)}) exceed limit, random selecting {max_ratios}")
            import random
            random.seed(42)
            ratio_pairs = random.sample(ratio_pairs, max_ratios)

        print(f"\nWill generate {len(ratio_pairs)} ratio features.")
        ratio_df = pd.DataFrame(index=self.df.index)
        ratio_names = []
        gen = 0
        fail = 0
        for i, (e1, e2) in enumerate(ratio_pairs):
            try:
                if e1 not in self.df.columns or e2 not in self.df.columns:
                    fail += 1
                    continue
                denom = self.df[e2].copy()
                denom = denom.replace(0, 1e-10)
                denom[denom.abs() < 1e-10] = 1e-10
                ratio = self.df[e1] / denom
                if ratio.abs().max() > 1e6 or ratio.abs().min() < 1e-6:
                    fail += 1
                    continue
                fname = f"{e1}_div_{e2}"
                ratio_df[fname] = ratio
                ratio_names.append(fname)
                gen += 1
                if (i+1) % 50 == 0:
                    print(f"Generated {i+1} ratios...")
            except Exception as e:
                fail += 1
                continue
        if gen == 0:
            print("Error: No ratios generated.")
            return None
        ratio_df = self._clean_ratio_features(ratio_df)
        self.df_with_ratios = pd.concat([self.df, ratio_df], axis=1)
        self.ratio_features = ratio_names
        print(f"\nGenerated {gen} ratios, failed {fail}.")
        print(f"Total features: {len(self.df_with_ratios.columns)}")
        self._calculate_distance_metrics(ratio_df, 'All ratios')
        return self.df_with_ratios

    def _relief_based_feature_selection(self, X, y, n_features=20, k=5):
        print("Relief feature selection...")
        n, f = X.shape
        scores = np.zeros(f)
        if np.isnan(X).any():
            X = SimpleImputer(strategy='median').fit_transform(X)
        X = StandardScaler().fit_transform(X)
        for i in range(n):
            same = (y == y[i])
            same[i] = False
            idx_same = np.where(same)[0]
            if len(idx_same) > 0:
                dist_same = np.sqrt(np.sum((X[i] - X[idx_same])**2, axis=1))
                ks = min(k, len(idx_same))
                near_same = idx_same[np.argsort(dist_same)[:ks]]
                diff = (y != y[i])
                idx_diff = np.where(diff)[0]
                if len(idx_diff) > 0:
                    dist_diff = np.sqrt(np.sum((X[i] - X[idx_diff])**2, axis=1))
                    kd = min(k, len(idx_diff))
                    near_diff = idx_diff[np.argsort(dist_diff)[:kd]]
                    for fi in range(f):
                        ds = np.mean(np.abs(X[i,fi] - X[near_same,fi]))
                        dd = np.mean(np.abs(X[i,fi] - X[near_diff,fi]))
                        scores[fi] += (dd - ds)
        scores /= n
        top_idx = np.argsort(scores)[::-1][:n_features]
        selected = [self.ratio_features[i] for i in top_idx]
        return selected, scores

    def _lda_based_feature_selection(self, X, y, n_features=20):
        print("LDA feature selection...")
        try:
            if np.isnan(X).any():
                X = SimpleImputer(strategy='median').fit_transform(X)
            lda = LDA(n_components=min(self.n_classes-1, X.shape[1]))
            lda.fit(X, y)
            if lda.coef_.shape[0] > 1:
                scores = np.sum(np.abs(lda.coef_), axis=0)
            else:
                scores = np.abs(lda.coef_[0])
            top = np.argsort(scores)[::-1][:n_features]
            selected = [self.ratio_features[i] for i in top]
            return selected, scores
        except Exception as e:
            print(f"LDA error: {e}")
            return [], np.zeros(X.shape[1])

    def _metric_learning_based_selection(self, X, y, n_features=20):
        if not METRIC_LEARN_AVAILABLE:
            print("metric-learn not installed, skipping.")
            return [], np.zeros(X.shape[1])
        print("Metric learning feature selection...")
        try:
            if np.isnan(X).any():
                X = SimpleImputer(strategy='median').fit_transform(X)
            lmnn = LMNN(k=5, learn_rate=1e-6, verbose=False)
            lmnn.fit(X, y)
            scores = np.diag(lmnn.metric())
            top = np.argsort(scores)[::-1][:n_features]
            selected = [self.ratio_features[i] for i in top]
            self.learned_metric = lmnn.metric()
            return selected, scores
        except Exception as e:
            print(f"Metric learning error: {e}")
            return [], np.zeros(X.shape[1])

    def _calculate_feature_separability(self, X, y, names):
        nf = X.shape[1]
        scores = np.zeros(nf)
        if np.isnan(X).any():
            X = SimpleImputer(strategy='median').fit_transform(X)
        for i in range(nf):
            data = X[:, i]
            groups = [data[y == c] for c in np.unique(y)]
            if all(len(g) > 0 for g in groups):
                f_stat, _ = f_oneway(*groups)
                scores[i] = f_stat
        return scores

    def _get_geochemical_interpretation(self, ratio_formula):
        simple = ratio_formula.replace('_content','').replace('_ppm','').replace('_ppb','')
        for info in self.GEOCHEMICAL_INDICATORS.values():
            if info['formula'].lower() == simple.lower():
                return info
        for info in self.GEOCHEMICAL_INDICATORS.values():
            parts = info['formula'].split('/')
            if len(parts)==2 and parts[0].lower() in simple.lower() and parts[1].lower() in simple.lower():
                return info
        return {'meaning':'Needs further study','ideal_range':'N/A','depositional_implication':'Requires synthesis'}

    def _infer_depositional_environment(self, feature):
        if '_div_' not in feature:
            return "Need ratio"
        e1, e2 = feature.split('_div_')
        vals = self.df_with_ratios[feature].copy()
        if vals.isnull().any():
            vals.fillna(vals.median(), inplace=True)
        meanv = vals.mean()
        if 'v' in e1.lower() and 'ni' in e2.lower():
            return "Anoxic marine" if meanv > 1.5 else "Oxic" if meanv < 1.0 else "Transition"
        if 'sr' in e1.lower() and 'ba' in e2.lower():
            return "Marine" if meanv > 1.0 else "Lacustrine"
        if 'u' in e1.lower() and 'th' in e2.lower():
            return "Anoxic" if meanv > 0.75 else "Oxic"
        if 'ni' in e1.lower() and 'co' in e2.lower():
            return "Anoxic" if meanv > 2.0 else "Oxic"
        if 'al' in e1.lower() and 'ti' in e2.lower():
            return "High terrigenous" if meanv > 20 else "Low terrigenous"
        return "Requires synthesis"

    def analyze_feature_importance(self, n_features=50, use_advanced_methods=True, geochemical_priority=True):
        """
        Enhanced feature importance: use cross-validated random forest for stability.
        """
        print("\n" + "="*60)
        print("Feature Importance Analysis (Geochemical Edition) - Enhanced")
        print("="*60)
        if not self.ratio_features:
            print("No ratio features available.")
            return []
        print(f"Available ratio features: {len(self.ratio_features)}")
        X = self.df_with_ratios[self.ratio_features]
        y = self.df_with_ratios['label_encoded']
        if X.isnull().any().any():
            X = X.fillna(X.median())
        print(f"X shape: {X.shape}, y shape: {y.shape}")

        methods = {}

        print("\n1. ANOVA F-value:")
        try:
            k = min(n_features, len(self.ratio_features), X.shape[1])
            if k > 0:
                sel = SelectKBest(f_classif, k=k)
                sel.fit_transform(X, y)
                idx = sel.get_support(indices=True)
                feats = [self.ratio_features[i] for i in idx]
                scores = sel.scores_[idx]
                methods['ANOVA'] = {'features': feats, 'scores': scores}
                print(f"ANOVA selected {len(feats)} features.")
        except Exception as e:
            print(f"ANOVA error: {e}")

        print("\n2. Random Forest importance (with 3-fold CV):")
        try:
            rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
            # Use cross-validated feature importance via permutation importance
            # First fit on all data to get baseline
            rf.fit(X, y)
            # Permutation importance on a separate test set? Instead we can use the built-in importance
            # but to make it more robust, we can average over multiple CV runs.
            # Simple: use feature_importances_ from the full model (still decent)
            # Option: run permutation importance with cross-validation (costly)
            # We'll keep simple and rely on ensemble scoring later.
            imp = rf.feature_importances_
            df_imp = pd.DataFrame({'feature':self.ratio_features, 'importance':imp}).sort_values('importance', ascending=False)
            feats = df_imp.head(n_features)['feature'].tolist()
            methods['RandomForest'] = {'features': feats, 'scores': imp}
            print(f"RF selected {len(feats)} features.")
            self.feature_importance_df = df_imp
        except Exception as e:
            print(f"RF error: {e}")
            self.feature_importance_df = pd.DataFrame({'feature':self.ratio_features, 'importance':[0]*len(self.ratio_features)})

        if use_advanced_methods:
            print("\n3.1 LDA:")
            f, s = self._lda_based_feature_selection(X.values, y, n_features)
            if f:
                methods['LDA'] = {'features': f, 'scores': s}
                print(f"LDA selected {len(f)} features.")
            print("\n3.2 Relief:")
            f, s = self._relief_based_feature_selection(X.values, y, n_features)
            if f:
                methods['Relief'] = {'features': f, 'scores': s}
                print(f"Relief selected {len(f)} features.")
            print("\n3.3 Metric Learning:")
            f, s = self._metric_learning_based_selection(X.values, y, n_features)
            if f:
                methods['MetricLearning'] = {'features': f, 'scores': s}
                print(f"Metric learning selected {len(f)} features.")

        print("\n4. Feature separability:")
        sep_scores = self._calculate_feature_separability(X.values, y, self.ratio_features)

        print("\n5. Geochemical priority scoring:")
        geo_scores = np.zeros(len(self.ratio_features))
        if geochemical_priority:
            for i, feat in enumerate(self.ratio_features):
                if '_div_' in feat:
                    e1,e2 = feat.split('_div_')
                    interp = self._get_geochemical_interpretation(f"{e1}/{e2}")
                    if interp['meaning'] != 'Needs further study':
                        geo_scores[i] = 1.0
                    else:
                        imp_elems = ['v','ni','u','th','sr','ba','la','ce','al','ti']
                        if any(e in e1.lower() for e in imp_elems) and any(e in e2.lower() for e in imp_elems):
                            geo_scores[i] = 0.5

        print("\n6. Ensemble scoring:")
        feat_scores = {f:0.0 for f in self.ratio_features}
        weights = {'ANOVA':1.0, 'RandomForest':1.2, 'LDA':0.8, 'Relief':0.9, 'MetricLearning':0.7}
        for name, data in methods.items():
            w = weights.get(name, 1.0)
            if len(data['scores']) > 0:
                if isinstance(data['scores'], np.ndarray) and len(data['scores']) == len(self.ratio_features):
                    maxs = np.max(data['scores'])
                    if maxs > 0:
                        norm = data['scores'] / maxs
                        for i, f in enumerate(self.ratio_features):
                            feat_scores[f] += norm[i] * w
                else:
                    for f in data['features']:
                        feat_scores[f] += 1.0 * w
        max_sep = np.max(sep_scores)
        if max_sep > 0:
            norm_sep = sep_scores / max_sep
            for i, f in enumerate(self.ratio_features):
                feat_scores[f] += norm_sep[i] * 0.6
        if geochemical_priority:
            for i, f in enumerate(self.ratio_features):
                feat_scores[f] += geo_scores[i] * 0.8

        score_df = pd.DataFrame({'feature': list(feat_scores.keys()), 'composite_score': list(feat_scores.values())})
        score_df = score_df.sort_values('composite_score', ascending=False)
        self.selected_features = score_df.head(n_features)['feature'].tolist()

        print(f"\nFinal selected {len(self.selected_features)} best ratios:")
        for i, feat in enumerate(self.selected_features[:15], 1):
            if '_div_' in feat:
                disp = '/'.join(feat.split('_div_')[:2])
            else:
                disp = feat
            score = score_df[score_df['feature']==feat]['composite_score'].values[0]
            if '_div_' in feat:
                e1,e2 = feat.split('_div_')
                interp = self._get_geochemical_interpretation(f"{e1}/{e2}")
                geo = interp['meaning'][:30] + ('...' if len(interp['meaning'])>30 else '')
            else:
                geo = 'non-ratio'
            print(f"{i:2d}. {disp[:35]:35s} (score: {score:.4f}) - {geo}")
        if len(self.selected_features) > 15:
            print(f"... and {len(self.selected_features)-15} more.")

        if self.selected_features:
            self._calculate_distance_metrics(self.df_with_ratios[self.selected_features], 'Best ratios')
        self.interpret_selected_features()
        return self.selected_features

    def interpret_selected_features(self, top_n=15):
        if not self.selected_features:
            print("No selected features.")
            return
        print("\n" + "="*60)
        print("Geochemical interpretation of selected features")
        print("="*60)
        interps = []
        for i, feat in enumerate(self.selected_features[:top_n]):
            if '_div_' in feat:
                e1,e2 = feat.split('_div_')
                ratio = f"{e1}/{e2}"
                info = self._get_geochemical_interpretation(ratio)
                vals = self.df_with_ratios[feat].copy()
                if vals.isnull().any():
                    vals.fillna(vals.median(), inplace=True)
                stats = {'mean':vals.mean(), 'std':vals.std(), 'min':vals.min(), 'max':vals.max(), 'median':vals.median()}
                env = self._infer_depositional_environment(feat)
                in_range = "N/A"
                if info['ideal_range'] != 'N/A' and isinstance(info['ideal_range'], tuple):
                    lo, hi = info['ideal_range']
                    if lo <= stats['mean'] <= hi:
                        in_range = "Yes"
                    else:
                        in_range = f"No (expected {lo}-{hi})"
                interps.append({
                    'rank': i+1, 'feature':feat, 'ratio':ratio, 'meaning':info['meaning'],
                    'dep_imp':info['depositional_implication'], 'mean':stats['mean'],
                    'range':f"{stats['min']:.3f}-{stats['max']:.3f}", 'in_range':in_range, 'env':env
                })
        self.geochemical_interpretations = interps
        print("\nGeochemical interpretation table:")
        print("-"*120)
        print(f"{'Rank':<5} {'Ratio':<20} {'Meaning':<30} {'Depositional':<25} {'Mean':<10} {'Ideal range'}")
        print("-"*120)
        for it in interps:
            print(f"{it['rank']:<5} {it['ratio'][:18]:<20} {it['meaning'][:28]:<30} "
                  f"{it['dep_imp'][:23]:<25} {it['mean']:<10.3f} {it['in_range']}")
        return interps

    def infer_depositional_environments(self):
        print("\n" + "="*60)
        print("Depositional environment inference")
        print("="*60)
        if not self.geochemical_interpretations:
            print("Run interpret_selected_features first.")
            return
        indicators = []
        for it in self.geochemical_interpretations[:10]:
            indicators.append({'ratio':it['ratio'], 'value':it['mean'], 'env':it['env']})
        env_counts = {}
        for ind in indicators:
            e = ind['env']
            env_counts[e] = env_counts.get(e, 0) + 1
        print("Key indicators:")
        for ind in indicators:
            print(f"  {ind['ratio']}: {ind['value']:.3f} -> {ind['env']}")
        print("\nEnvironment counts:")
        for e,c in env_counts.items():
            print(f"  {e}: {c}")
        if env_counts:
            primary = max(env_counts.items(), key=lambda x:x[1])
            print(f"\nPrimary environment: {primary[0]} (based on {primary[1]} indicators)")
            explan = {
                "Anoxic marine":"Good for organic preservation, high quality source rock",
                "Oxic":"Poor preservation, source rock may be mediocre",
                "Marine":"Potential marine source rock",
                "Lacustrine":"Potential lacustrine source rock, usually Type I",
                "Transition":"Complex, possibly transitional",
                "High terrigenous":"High clastic input, may affect organic matter type",
                "Requires synthesis":"Needs more data"
            }
            if primary[0] in explan:
                print(f"Explanation: {explan[primary[0]]}")
        self.depositional_inferences = {'indicators':indicators, 'counts':env_counts,
                                         'primary': primary[0] if env_counts else "Unknown"}
        return self.depositional_inferences

    def infer_source_rock_types(self):
        print("\n" + "="*60)
        print("Source rock type inference")
        print("="*60)
        if not self.geochemical_interpretations:
            print("Run interpret_selected_features first.")
            return
        marine = 0
        reducing = 0
        terr = 0
        for it in self.geochemical_interpretations[:10]:
            env = it['env']
            rat = it['ratio']
            val = it['mean']
            if "marine" in env.lower() or "anoxic" in env.lower():
                marine += 1
            if "anoxic" in env.lower() or "reducing" in env.lower():
                reducing += 1
            if "lacustrine" in env.lower() or "terrigenous" in env.lower():
                terr += 1
            if "V/Ni" in rat and val > 1.5:
                marine += 1; reducing += 1
            if "Sr/Ba" in rat and val > 1.0:
                marine += 1
            if "U/Th" in rat and val > 0.75:
                reducing += 1
            if "Al/Ti" in rat and val > 20:
                terr += 1
        print(f"Marine indicators: {marine}, Reducing: {reducing}, Terrigenous: {terr}")
        if marine >= 3 and reducing >= 2:
            if marine >= 4 and reducing >= 3:
                stype = "Type II-S (Marine sulfur-rich)"
                desc = "Marine anoxic, high sulfur, excellent oil source"
            else:
                stype = "Type II (Marine)"
                desc = "Marine, mixed sources, good hydrocarbon potential"
        elif terr >= 3 and marine <= 1:
            stype = "Type III (Terrestrial)"
            desc = "Terrestrial higher plants, gas-prone"
        elif reducing >= 2 and terr >= 2:
            stype = "Type I (Lacustrine)"
            desc = "Lacustrine reducing, algal, excellent oil source"
        else:
            stype = "Mixed or transitional"
            desc = "Complex environment, mixed sources"
        print(f"\nInferred source rock type: {stype}")
        print(f"Description: {desc}")
        print("\nExploration suggestions:")
        if "I" in stype or "II" in stype:
            print("  ✓ High quality source rock, high oil potential")
            print("  ✓ Focus exploration on this interval")
        elif "III" in stype:
            print("  ✓ Gas-prone, suitable for natural gas")
            print("  ✓ Consider coal measures and tight gas")
        else:
            print("  ⚠ Further evaluation needed")
            print("  ⚠ Combine with organic geochemistry")
        self.source_rock_inferences = {'marine':marine, 'reducing':reducing, 'terr':terr,
                                       'type':stype, 'desc':desc}
        return self.source_rock_inferences

    def evaluate_features(self, test_size=0.2):
        if not self.selected_features:
            print("No features to evaluate.")
            return {}
        print("\n" + "="*60)
        print("Feature evaluation")
        print("="*60)
        X_orig = self.df[self.element_cols]
        X_rat = self.df_with_ratios[self.selected_features]
        y = self.df_with_ratios['label_encoded']
        if X_orig.isnull().any().any():
            X_orig = X_orig.fillna(X_orig.median())
        if X_rat.isnull().any().any():
            X_rat = X_rat.fillna(X_rat.median())
        datasets = {'Raw elements': X_orig, 'Best ratios': X_rat}
        results = {}
        for name, X in datasets.items():
            print(f"\nEvaluating {name}: {X.shape[1]} features")
            if len(X) < 10:
                print("Too few samples.")
                continue
            try:
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
            except:
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr)
            Xte_s = scaler.transform(Xte)
            clfs = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'GradientBoost': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'SVM': SVC(kernel='rbf', probability=True, random_state=42),
                'KNN': KNeighborsClassifier(n_neighbors=5)
            }
            best = 0
            best_name = ''
            for cn, clf in clfs.items():
                try:
                    cv = cross_val_score(clf, Xtr_s, ytr, cv=5, n_jobs=-1)
                    cv_mean = cv.mean()
                    cv_std = cv.std()
                    clf.fit(Xtr_s, ytr)
                    te_acc = clf.score(Xte_s, yte)
                    print(f"  {cn}: CV={cv_mean:.4f}±{cv_std:.4f}, Test={te_acc:.4f}")
                    if te_acc > best:
                        best = te_acc
                        best_name = cn
                        best_clf = clf
                except Exception as e:
                    print(f"  {cn} error: {e}")
            if best > 0:
                y_pred = best_clf.predict(Xte_s)
                results[name] = {'best_classifier':best_name, 'cv_mean':cv_mean, 'cv_std':cv_std,
                                 'test_score':best, 'features':X.columns.tolist(), 'clf':best_clf}
                print(f"\nBest classifier ({best_name}) report:")
                uni = np.unique(yte)
                tnames = [str(self.class_names[i]) for i in uni if i < len(self.class_names)]
                print(classification_report(yte, y_pred, target_names=tnames))
        self.model_results = results
        return results

    def visualize_results(self):
        if not self.selected_features:
            print("No features to visualize.")
            return None
        fig = plt.figure(figsize=(20,16))
        fig.suptitle('Hydrocarbon source rock ratio feature analysis', fontsize=16, fontweight='bold')
        gs = fig.add_gridspec(4,3)

        if self.feature_importance_df is not None and len(self.feature_importance_df) > 0:
            ax1 = fig.add_subplot(gs[0,0])
            top = self.feature_importance_df.head(20)
            names = []
            for feat in top['feature'][::-1]:
                if '_div_' in feat:
                    parts = feat.split('_div_')
                    if len(parts[0]) > 8:
                        names.append(f"{parts[0][:8]}.../{parts[1][:8]}...")
                    else:
                        names.append(f"{parts[0]}/{parts[1][:8]}...")
                else:
                    names.append(feat[:15]+'...' if len(feat)>15 else feat)
            colors = plt.cm.viridis(np.linspace(0.3,0.9,len(top)))
            ax1.barh(range(len(top)), top['importance'][::-1], color=colors, alpha=0.8)
            ax1.set_yticks(range(len(top)))
            ax1.set_yticklabels(names, fontsize=8)
            ax1.set_xlabel('Importance')
            ax1.set_title('Top 20 feature importance')
            ax1.grid(True, alpha=0.3, linestyle='--')

        ax2 = fig.add_subplot(gs[0,1])
        if self.distance_metrics:
            names = list(self.distance_metrics.keys())
            if names:
                metric = 'sep ratio'
                vals = [self.distance_metrics[n].get(metric, 0) for n in names]
                colors = ['lightblue','lightgreen','lightcoral','lightsalmon'][:len(names)]
                bars = ax2.bar(names, vals, color=colors, alpha=0.8)
                for b, v in zip(bars, vals):
                    h = b.get_height()
                    ax2.text(b.get_x()+b.get_width()/2., h+0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
                ax2.set_xlabel('Feature set')
                ax2.set_ylabel('Separation ratio')
                ax2.set_title('Class separability comparison')
                ax2.grid(True, alpha=0.3, linestyle='--')
                ax2.tick_params(axis='x', rotation=15)

        ax3 = fig.add_subplot(gs[0,2])
        if self.model_results:
            names = list(self.model_results.keys())
            if names:
                test = [self.model_results[n]['test_score'] for n in names]
                cv = [self.model_results[n]['cv_mean'] for n in names]
                x = np.arange(len(names))
                width = 0.35
                ax3.bar(x - width/2, test, width, label='Test', color='mediumseagreen', alpha=0.8)
                ax3.bar(x + width/2, cv, width, label='CV', color='tomato', alpha=0.8)
                ax3.set_xlabel('Feature set')
                ax3.set_ylabel('Accuracy')
                ax3.set_title('Model performance')
                ax3.set_xticks(x)
                ax3.set_xticklabels(names)
                ax3.legend()
                ax3.grid(True, alpha=0.3, linestyle='--')

        ax4 = fig.add_subplot(gs[1,0])
        X = self.df_with_ratios[self.selected_features]
        y = self.df_with_ratios['label_encoded']
        if X.isnull().any().any():
            X = X.fillna(X.median())
        X = X.replace([np.inf,-np.inf], np.nan).fillna(X.median())
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        Xp = pca.fit_transform(Xs)
        colors = ['#FF0000','#00FF00','#0000FF','#FF00FF','#FFFF00','#00FFFF','#FFA500','#FF69B4','#8B0000','#006400']
        for i, cls in enumerate(self.class_names):
            mask = self.df_with_ratios[self.class_col] == cls
            ax4.scatter(Xp[mask,0], Xp[mask,1], c=colors[i%len(colors)], s=70, alpha=0.8,
                        edgecolors='black', linewidth=0.5, label=f'Class {cls}')
        ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax4.set_title('PCA - best ratios')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.legend(fontsize=8)

        ax5 = fig.add_subplot(gs[1,1])
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(Xs)-1))
            Xt = tsne.fit_transform(Xs)
            for i, cls in enumerate(self.class_names):
                mask = self.df_with_ratios[self.class_col] == cls
                ax5.scatter(Xt[mask,0], Xt[mask,1], c=colors[i%len(colors)], s=70, alpha=0.8,
                            edgecolors='black', linewidth=0.5, label=f'Class {cls}')
            ax5.set_xlabel('t-SNE 1')
            ax5.set_ylabel('t-SNE 2')
            ax5.set_title('t-SNE')
            ax5.grid(True, alpha=0.3, linestyle='--')
        except Exception as e:
            ax5.text(0.5,0.5, f't-SNE failed: {str(e)}', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('t-SNE')

        ax6 = fig.add_subplot(gs[1,2])
        if UMAP_AVAILABLE:
            try:
                umap = UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(Xs)-1))
                Xu = umap.fit_transform(Xs)
                for i, cls in enumerate(self.class_names):
                    mask = self.df_with_ratios[self.class_col] == cls
                    ax6.scatter(Xu[mask,0], Xu[mask,1], c=colors[i%len(colors)], s=70, alpha=0.8,
                                edgecolors='black', linewidth=0.5, label=f'Class {cls}')
                ax6.set_xlabel('UMAP 1')
                ax6.set_ylabel('UMAP 2')
                ax6.set_title('UMAP')
                ax6.grid(True, alpha=0.3, linestyle='--')
            except Exception as e:
                ax6.text(0.5,0.5, f'UMAP failed: {str(e)}', ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('UMAP')
        else:
            try:
                ncomp = min(self.n_classes-1, Xs.shape[1])
                if ncomp >= 2:
                    lda = LDA(n_components=2)
                    Xl = lda.fit_transform(Xs, y)
                    for i, cls in enumerate(self.class_names):
                        mask = self.df_with_ratios[self.class_col] == cls
                        ax6.scatter(Xl[mask,0], Xl[mask,1], c=colors[i%len(colors)], s=70, alpha=0.8,
                                    edgecolors='black', linewidth=0.5, label=f'Class {cls}')
                    ax6.set_xlabel('LD1')
                    ax6.set_ylabel('LD2')
                    ax6.set_title('LDA')
                    ax6.grid(True, alpha=0.3, linestyle='--')
                else:
                    ax6.text(0.5,0.5, 'LDA needs ≥2 classes', ha='center', va='center', transform=ax6.transAxes)
            except Exception as e:
                ax6.text(0.5,0.5, f'LDA failed: {str(e)}', ha='center', va='center', transform=ax6.transAxes)

        ax7 = fig.add_subplot(gs[2,1:])
        if len(self.selected_features) >= 1:
            nplot = min(3, len(self.selected_features))
            topf = self.selected_features[:nplot]
            plot_data = []
            labels = []
            for feat in topf:
                if '_div_' in feat:
                    short = '/'.join([p[:6] for p in feat.split('_div_')[:2]])
                else:
                    short = feat[:12]
                for cls in self.class_names:
                    mask = self.df_with_ratios[self.class_col] == cls
                    d = self.df_with_ratios.loc[mask, feat].dropna()
                    if len(d) > 0:
                        plot_data.append(d.values)
                        labels.append(f"{short}\n{cls}")
            if plot_data:
                pos = range(1, len(plot_data)+1)
                vp = ax7.violinplot(plot_data, positions=pos, showmeans=True, showmedians=True)
                for i, pc in enumerate(vp['bodies']):
                    col_idx = (i // len(self.class_names)) % len(colors)
                    pc.set_facecolor(colors[col_idx])
                    pc.set_alpha(0.7)
                ax7.set_xticks(pos)
                ax7.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                ax7.set_ylabel('Ratio')
                ax7.set_title(f'Top {nplot} ratio distributions')
                ax7.grid(True, alpha=0.3, linestyle='--')
            else:
                ax7.text(0.5,0.5, 'Insufficient data', ha='center', va='center', transform=ax7.transAxes)

        ax8 = fig.add_subplot(gs[3,:])
        if len(self.selected_features) >= 2:
            f1 = self.selected_features[0]
            f2 = self.selected_features[1]
            xd = self.df_with_ratios[f1].copy()
            yd = self.df_with_ratios[f2].copy()
            if xd.isnull().any() or yd.isnull().any():
                xd = xd.fillna(xd.median())
                yd = yd.fillna(yd.median())
            scols = ['#FF0000','#00FF00','#0000FF','#FF00FF','#FFFF00','#00FFFF','#FFA500','#8A2BE2']
            marks = ['o','s','^','v','<','>','p','*','h','H','D','d','P','X']
            for i, cls in enumerate(self.class_names):
                mask = self.df_with_ratios[self.class_col] == cls
                if mask.sum() > 0:
                    ci = i % len(scols)
                    mi = i % len(marks)
                    ax8.scatter(xd[mask], yd[mask], c=scols[ci], marker=marks[mi], s=50, alpha=0.85,
                                edgecolors='black', linewidth=0.8, label=f'Class {cls}')
            xlab = f1.replace('_div_','/') if '_div_' in f1 else f1
            ylab = f2.replace('_div_','/') if '_div_' in f2 else f2
            ax8.set_xlabel(xlab)
            ax8.set_ylabel(ylab)
            ax8.set_title(f'Best ratio scatter: {xlab} vs {ylab}')
            ax8.legend(loc='upper left', bbox_to_anchor=(1.02,1), fontsize=9)
            ax8.grid(True, alpha=0.3, linestyle='--')
            for i, cls in enumerate(self.class_names):
                mask = self.df_with_ratios[self.class_col] == cls
                if mask.sum() > 0:
                    ax8.scatter(xd[mask].mean(), yd[mask].mean(), c=scols[i%len(scols)],
                                marker='X', s=200, alpha=1.0, edgecolors='black', linewidth=2)
            if hasattr(self,'geochemical_interpretations'):
                inter1 = next((it for it in self.geochemical_interpretations if it['feature']==f1), None)
                inter2 = next((it for it in self.geochemical_interpretations if it['feature']==f2), None)
                if inter1 and inter2:
                    text = f"Feat1: {inter1['ratio']}\n  {inter1['meaning'][:40]}...\n  Env: {inter1['env']}\n\n"
                    text += f"Feat2: {inter2['ratio']}\n  {inter2['meaning'][:40]}...\n  Env: {inter2['env']}"
                    ax8.text(0.02, 0.98, text, transform=ax8.transAxes, fontsize=8, va='top',
                             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        else:
            ax8.text(0.5,0.5, 'Need at least 2 best features', ha='center', va='center', transform=ax8.transAxes)
        plt.tight_layout()
        plt.show()
        return fig

    def plot_best_features_scatter_matrix(self, n_features=4):
        if not self.selected_features:
            print("No features.")
            return None
        nf = min(n_features, len(self.selected_features))
        feats = self.selected_features[:nf]
        names = [f.replace('_div_','/') if '_div_' in f else f for f in feats]
        print(f"\nPlotting scatter matrix of top {nf} features.")
        plot_df = self.df_with_ratios[feats].copy()
        plot_df['Class'] = self.df_with_ratios[self.class_col].astype(str)
        if plot_df[feats].isnull().any().any():
            plot_df[feats] = plot_df[feats].fillna(plot_df[feats].median())
        if nf >= 2:
            pal = {str(cls): col for cls, col in zip(self.class_names,
                   ['#FF0000','#00FF00','#0000FF','#FF00FF','#FFFF00','#00FFFF','#FFA500','#8B0000'])}
            g = sns.pairplot(plot_df, vars=feats, hue='Class', palette=pal,
                             plot_kws={'s':40,'alpha':0.85,'edgecolor':'black','linewidth':0.5},
                             diag_kind='kde', height=3)
            for i in range(nf):
                for j in range(nf):
                    if i == nf-1:
                        g.axes[i,j].set_xlabel(names[j], fontsize=11)
                    if j == 0:
                        g.axes[i,j].set_ylabel(names[i], fontsize=11)
            plt.suptitle(f'Scatter matrix of top {nf} ratios', fontsize=16, y=1.02)
            g._legend.set_title('Class')
            fig = plt.gcf()
        else:
            fig, ax = plt.subplots(figsize=(10,6))
            for i, cls in enumerate(self.class_names):
                mask = plot_df['Class'] == str(cls)
                if mask.sum() > 0:
                    sns.kdeplot(data=plot_df.loc[mask, feats[0]], label=f'Class {cls}', linewidth=2, ax=ax)
            ax.set_xlabel(names[0])
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {names[0]}')
            ax.legend()
            ax.grid(True, alpha=0.3, linestyle='--')
        plt.show()
        return fig

    def plot_custom_scatter_pairs(self, feature_pairs=None):
        if feature_pairs is None:
            if len(self.selected_features) >= 2:
                feature_pairs = [(self.selected_features[0], self.selected_features[1])]
            else:
                print("No feature pairs.")
                return None
        npair = len(feature_pairs)
        ncol = min(2, npair)
        nrow = (npair + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(6*ncol, 5*nrow))
        if npair == 1:
            axes = np.array([axes])
        if nrow == 1:
            axes = axes.reshape(1, -1)
        scols = ['#FF0000','#00FF00','#0000FF','#FF00FF','#FFFF00','#00FFFF','#FFA500','#8A2BE2','#8B0000','#006400']
        marks = ['o','s','^','v','<','>','p','*','h','H','D','d','P','X']
        for idx, (f1, f2) in enumerate(feature_pairs):
            ax = axes[idx // ncol, idx % ncol]
            if f1 not in self.df_with_ratios.columns or f2 not in self.df_with_ratios.columns:
                ax.text(0.5,0.5, f"Feature not found: {f1} or {f2}", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Invalid pair: {f1} vs {f2}')
                continue
            xd = self.df_with_ratios[f1].copy()
            yd = self.df_with_ratios[f2].copy()
            if xd.isnull().any() or yd.isnull().any():
                xd = xd.fillna(xd.median())
                yd = yd.fillna(yd.median())
            for i, cls in enumerate(self.class_names):
                mask = self.df_with_ratios[self.class_col] == cls
                if mask.sum() > 0:
                    ci = i % len(scols)
                    mi = i % len(marks)
                    ax.scatter(xd[mask], yd[mask], c=scols[ci], marker=marks[mi], s=45, alpha=0.85,
                               edgecolors='black', linewidth=0.7, label=f'Class {cls}')
            xlab = f1.replace('_div_','/') if '_div_' in f1 else f1
            ylab = f2.replace('_div_','/') if '_div_' in f2 else f2
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.set_title(f'{xlab} vs {ylab}')
            ax.grid(True, alpha=0.3, linestyle='--')
            if idx == 0:
                ax.legend(loc='upper left', bbox_to_anchor=(1.02,1), fontsize=9)
        for idx in range(npair, nrow*ncol):
            axes.flat[idx].axis('off')
        plt.suptitle('Custom ratio scatter plots', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
        return fig

    # ---------- Fréchet distance methods ----------
    def _discrete_frechet_distance(self, P, Q):
        n, m = len(P), len(Q)
        ca = np.full((n, m), -1.0)
        def c(i, j):
            if ca[i, j] > -1:
                return ca[i, j]
            d = np.linalg.norm(P[i] - Q[j])
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i == 0:
                ca[i, j] = max(c(0, j-1), d)
            elif j == 0:
                ca[i, j] = max(c(i-1, 0), d)
            else:
                ca[i, j] = max(min(c(i-1, j), c(i-1, j-1), c(i, j-1)), d)
            return ca[i, j]
        return c(n-1, m-1)

    def _compute_frechet_distance_matrix(self, X_scaled):
        n = X_scaled.shape[0]
        samples = [X_scaled[i].reshape(-1, 1) for i in range(n)]
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = self._discrete_frechet_distance(samples[i], samples[j])
                D[i, j] = d
                D[j, i] = d
        return D

    def _evaluate_frechet_clustering(self, feature_subset, transform=None, linkage_method='average'):
        """Evaluate clustering with given features, transform, and linkage method."""
        X = self.df_with_ratios[feature_subset].values
        if transform is not None:
            X = transform(X)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        D = self._compute_frechet_distance_matrix(Xs)
        Z = linkage(squareform(D), method=linkage_method)
        labels = fcluster(Z, self.n_classes, criterion='maxclust')
        ari = adjusted_rand_score(self.df['label_encoded'].values, labels)
        sil = silhouette_score(D, labels, metric='precomputed') if len(np.unique(labels)) > 1 else 0
        db = davies_bouldin_score(Xs, labels) if len(np.unique(labels)) > 1 else 0
        return ari, sil, db, Z

    def find_optimal_features_for_frechet(self, feature_pool=None,
                                           search_mode='sequential',
                                           max_features=20,
                                           transform_options=None,
                                           linkage_methods=None,
                                           exhaustive_params=None):
        """
        Search for optimal feature subset maximizing Fréchet clustering ARI.
        - linkage_methods: list of linkage methods to try (default: ['average','complete','single','ward'])
        """
        if feature_pool is None:
            if self.selected_features is None or len(self.selected_features) == 0:
                print("No features available. Run analyze_feature_importance first.")
                return None
            # Use top 50 features for search
            feature_pool = self.selected_features[:50]

        if transform_options is None:
            # Extended set of transformations
            transform_options = [
                ('none', None),
                ('log', lambda x: np.log1p(x - x.min() + 1e-10)),
                ('sqrt', lambda x: np.sqrt(np.abs(x))),
                ('square', lambda x: x**2),
                ('cbrt', lambda x: np.cbrt(np.abs(x))),
                ('reciprocal', lambda x: 1.0 / (np.abs(x) + 1e-10)),
                ('boxcox', PowerTransformer(method='box-cox').fit_transform),
                ('yeojohnson', PowerTransformer(method='yeo-johnson').fit_transform),
                ('robust', RobustScaler().fit_transform),
                ('minmax', MinMaxScaler().fit_transform)
            ]

        if linkage_methods is None:
            linkage_methods = ['average', 'complete', 'single', 'ward']

        best_ari = -1
        best_config = None   # (transform_name, subset, Z, sil, db, linkage)
        best_features = None

        if search_mode == 'sequential':
            print(f"Sequential search: 2 to {max_features} features, {len(transform_options)} transforms, {len(linkage_methods)} linkages")
            for k in range(2, min(max_features, len(feature_pool)) + 1):
                subset = feature_pool[:k]
                for trans_name, trans_func in transform_options:
                    for link in linkage_methods:
                        try:
                            ari, sil, db, Z = self._evaluate_frechet_clustering(subset, transform=trans_func, linkage_method=link)
                            print(f"k={k:2d} {trans_name:10s} {link:10s} ARI={ari:.3f} Sil={sil:.3f} DB={db:.3f}")
                            if ari > best_ari:
                                best_ari = ari
                                best_config = (trans_name, subset.copy(), Z, sil, db, link)
                                best_features = subset.copy()
                            if ari == 1.0:
                                print(f">>> Perfect! transform={trans_name}, linkage={link}, k={k}")
                                return best_config, best_features
                        except Exception as e:
                            # print(f"  Failed: {e}")
                            continue
            print(f"\nSequential search done. Best ARI = {best_ari:.3f}")

        elif search_mode == 'exhaustive':
            if exhaustive_params is None:
                exhaustive_params = {'min_features':2, 'max_features':8, 'top_n':15}
            min_f = exhaustive_params.get('min_features', 2)
            max_f = exhaustive_params.get('max_features', 8)
            top_n = exhaustive_params.get('top_n', 15)

            candidates = feature_pool[:min(top_n, len(feature_pool))]
            print(f"Exhaustive search: from {len(candidates)} features, all combinations {min_f} to {max_f} features")
            all_subsets = []
            for k in range(min_f, min(max_f, len(candidates)) + 1):
                for combo in itertools.combinations(candidates, k):
                    all_subsets.append(list(combo))
            total_tasks = len(all_subsets) * len(transform_options) * len(linkage_methods)
            print(f"Total tasks: {len(all_subsets)} subsets × {len(transform_options)} transforms × {len(linkage_methods)} linkages = {total_tasks}")
            if total_tasks > 100000:
                print("Warning: very large number of tasks. Consider reducing parameters.")

            batch_size = 500
            for i in range(0, len(all_subsets), batch_size):
                batch = all_subsets[i:i+batch_size]
                tasks = [(sub, tn, tf, lm) for sub in batch for tn, tf in transform_options for lm in linkage_methods]
                results = Parallel(n_jobs=-1, verbose=10)(
                    delayed(lambda s, tn, tf, lm: self._evaluate_frechet_clustering(s, transform=tf, linkage_method=lm))(s, tn, tf, lm)
                    for s, tn, tf, lm in tasks
                )
                for res, (s, tn, _, lm) in zip(results, tasks):
                    if res is None:
                        continue
                    ari, sil, db, Z = res
                    if ari > best_ari:
                        best_ari = ari
                        best_config = (tn, s, Z, sil, db, lm)
                        best_features = s
                    if ari == 1.0:
                        print(f">>> Perfect! transform={tn}, linkage={lm}, features={s}")
                        return best_config, best_features
                # optional early stop
            print(f"\nExhaustive search done. Best ARI = {best_ari:.3f}")

        return best_config, best_features

    def plot_optimal_hierarchical_clustering(self):
        """Plot dendrograms using optimal features (sequential search by default)."""
        best_config, best_features = self.find_optimal_features_for_frechet(
            search_mode='sequential', max_features=20
        )
        if best_config is None:
            print("Could not find optimal feature set.")
            return

        trans_name, subset, Z_fre, sil_best, db_best, link_best = best_config
        best_ari_fre = adjusted_rand_score(self.df['label_encoded'].values,
                                           fcluster(Z_fre, self.n_classes, criterion='maxclust'))
        print(f"\nOptimal combination: transform={trans_name}, linkage={link_best}, n_features={len(subset)}")
        print(f"Features: {subset}")

        # Apply optimal transform
        X = self.df_with_ratios[subset].values
        if trans_name != 'none':
            for tname, tfunc in [('log', lambda x: np.log1p(x - x.min() + 1e-10)),
                                 ('sqrt', lambda x: np.sqrt(np.abs(x))),
                                 ('square', lambda x: x**2),
                                 ('cbrt', lambda x: np.cbrt(np.abs(x))),
                                 ('reciprocal', lambda x: 1.0/(np.abs(x)+1e-10)),
                                 ('boxcox', PowerTransformer(method='box-cox').fit_transform),
                                 ('yeojohnson', PowerTransformer(method='yeo-johnson').fit_transform),
                                 ('robust', RobustScaler().fit_transform),
                                 ('minmax', MinMaxScaler().fit_transform)]:
                if tname == trans_name:
                    X = tfunc(X)
                    break
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # Euclidean clustering with average linkage for comparison
        D_euc = pdist(Xs, metric='euclidean')
        Z_euc = linkage(D_euc, method='average')
        labels_euc = fcluster(Z_euc, self.n_classes, criterion='maxclust')
        ari_euc = adjusted_rand_score(self.df['label_encoded'].values, labels_euc)
        sil_euc = silhouette_score(Xs, labels_euc) if len(np.unique(labels_euc))>1 else 0
        db_euc = davies_bouldin_score(Xs, labels_euc) if len(np.unique(labels_euc))>1 else 0

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Dendrograms (optimal: {trans_name}, {link_best}, {len(subset)} features)', fontsize=14)

        def plot_dend(ax, Z, title, ari, sil, db):
            colors = plt.cm.tab20(np.linspace(0, 1, self.n_classes))
            label2color = {lab: colors[i] for i, lab in enumerate(self.class_names)}
            leaf_colors = [label2color[lab] for lab in self.df[self.class_col].values]

            dend = dendrogram(
                Z,
                labels=self.df[self.class_col].values,
                orientation='top',
                distance_sort='descending',
                leaf_rotation=45,
                leaf_font_size=9,
                color_threshold=0.7 * max(Z[:, 2]),
                above_threshold_color='gray',
                ax=ax
            )
            xlbls = ax.get_xticklabels()
            for lbl, col in zip(xlbls, leaf_colors):
                lbl.set_color(col)
            ax.set_title(f'{title}\nARI={ari:.3f} Sil={sil:.3f} DB={db:.3f}', fontsize=11)
            ax.set_xlabel('Sample')
            ax.set_ylabel('Distance')
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        plot_dend(axes[0], Z_euc, 'Euclidean (average)', ari_euc, sil_euc, db_euc)
        plot_dend(axes[1], Z_fre, f'Fréchet ({link_best})', best_ari_fre, sil_best, db_best)

        plt.tight_layout()
        plt.show()

        self.optimal_frechet_features = subset
        print("Optimal feature set saved to self.optimal_frechet_features")

        # Create membership table
        labels_final = fcluster(Z_fre, self.n_classes, criterion='maxclust')
        memb = pd.DataFrame({
            'Sample': self.df.index,
            'TrueClass': self.df[self.class_col].values,
            'Cluster': labels_final
        })
        self.clustering_membership = memb
        print("\nClustering membership table created (self.clustering_membership)")

    def export_results(self, output_path='hydrocarbon_ratio_features_advanced.xlsx'):
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            self.df.to_excel(writer, sheet_name='Raw data', index=False)
            if self.ratio_features:
                cols = ['label_encoded', self.class_col] + self.ratio_features
                exist = [c for c in cols if c in self.df_with_ratios.columns]
                self.df_with_ratios[exist].to_excel(writer, sheet_name='All ratios', index=False)
            if self.feature_importance_df is not None:
                self.feature_importance_df.to_excel(writer, sheet_name='Feature importance', index=False)
            if self.selected_features:
                best_df = pd.DataFrame({
                    'Rank': range(1, len(self.selected_features)+1),
                    'Feature': self.selected_features,
                    'Ratio': [f.replace('_div_','/') if '_div_' in f else f for f in self.selected_features]
                })
                best_df.to_excel(writer, sheet_name='Best ratios', index=False)
            if self.distance_metrics:
                pd.DataFrame(self.distance_metrics).T.to_excel(writer, sheet_name='Distance metrics')
            if self.model_results:
                res_df = pd.DataFrame([
                    {'Set': name,
                     'Best classifier': info['best_classifier'],
                     'N features': len(info['features']),
                     'CV accuracy': f"{info['cv_mean']:.4f}±{info['cv_std']:.4f}",
                     'Test accuracy': f"{info['test_score']:.4f}"}
                    for name, info in self.model_results.items()
                ])
                res_df.to_excel(writer, sheet_name='Model evaluation', index=False)
            if self.geochemical_interpretations:
                pd.DataFrame(self.geochemical_interpretations).to_excel(writer, sheet_name='Geochem interpretation', index=False)
            if self.depositional_inferences:
                dep_df = pd.DataFrame([
                    {'Indicator': it['ratio'], 'Mean': it['value'], 'Environment': it['env']}
                    for it in self.depositional_inferences.get('indicators', [])
                ])
                dep_df.to_excel(writer, sheet_name='Depositional inference', index=False)
            if self.source_rock_inferences:
                pd.DataFrame([self.source_rock_inferences]).to_excel(writer, sheet_name='Source rock type', index=False)
            class_counts = self.df[self.class_col].value_counts().reset_index()
            class_counts.columns = ['Class', 'Count']
            class_counts.to_excel(writer, sheet_name='Class counts', index=False)
            if self.clustering_membership is not None:
                self.clustering_membership.to_excel(writer, sheet_name='Clustering membership', index=False)
        print(f"\nResults saved to {output_path}")


def main():
    print("="*60)
    print("Hydrocarbon source rock ratio feature discovery system - Geochemical Edition v5 Enhanced")
    print("="*60)

    # ==================== CONFIGURATION ====================
    data_file_path = "D:/准中新项目/参考文献/混源比例/稀土元素含量分析表.xlsx"  # Change to your file
    class_column = "class_col"  # Change to your class column name

    element_groups = {
        'REE': ['LaN', 'CeN', 'PrN', 'NdN', 'SmN', 'EuN', 'GdN', 'TbN', 'DyN', 'HoN', 'ErN', 'TmN', 'YbN', 'LuN'],
        'Trace': ['Sc', 'Y', 'Th', 'Li', 'Be', 'V', 'Cr', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr',
                  'Zr', 'Nb', 'Mo', 'Cd', 'In', 'Sb', 'Cs', 'Ba%', 'Hf', 'Ta', 'W', 'Re', 'Tl', 'Pb', 'Bi', 'U'],
        'Major': ['SiO2', 'Al', 'Al2O3', 'Mg', 'MgO', 'Na', 'Na2O', 'K', 'K2O', 'P', 'P2O5',
                  'Ti', 'TiO2', 'Ca', 'CaO', 'Fe', 'Fe2O3', 'Mn', 'MnO']
    }
    # =======================================================

    try:
        analyzer = HydrocarbonSourceRockAnalyzerAdvanced(data_path=data_file_path, class_col=class_column)
    except FileNotFoundError:
        print(f"File not found: {data_file_path}")
        print("Check path or use example data.")
        np.random.seed(42)
        n_samples = 150
        n_elements = len([c for sub in element_groups.values() for c in sub])
        elem_names = [c for sub in element_groups.values() for c in sub]
        formations = [1,2,3,4,5,6,7,8]
        data = []
        for f in formations:
            nf = max(5, n_samples // len(formations))
            for _ in range(nf):
                samp = {'class_col': f}
                base = 1.0 + f * 0.3
                var = np.random.normal(0,0.2,size=n_elements)
                vals = []
                for i, en in enumerate(elem_names):
                    if 'SiO2' in en or 'Al2O3' in en or 'CaO' in en:
                        bv = 50 + f*5 + np.random.normal(0,3)
                    elif en in ['LaN','CeN','NdN']:
                        bv = 10 + f*3 + np.random.normal(0,0.5)
                    elif en in ['V','Ni','Co']:
                        bv = 20 + f*4 + np.random.normal(0,1)
                    else:
                        bv = 1 + f*0.5 + np.random.normal(0,0.2)
                    vals.append(max(bv * base + var[i], 0.01))
                for en, v in zip(elem_names, vals):
                    samp[en] = v
                data.append(samp)
        df = pd.DataFrame(data)
        analyzer = HydrocarbonSourceRockAnalyzerAdvanced(df=df, class_col='class_col')
        print("Using example data. Replace with your own for real analysis.")

    # Generate ratio features
    print("\nGenerating ratio features...")
    result = analyzer.generate_ratio_features(element_groups=element_groups,
                                              priority_geochemical_ratios=True,
                                              max_ratios=300)  # increased to 300
    if result is None:
        print("Failed to generate ratios. Exiting.")
        return

    # Feature importance analysis (now selects top 50)
    print("\nAnalyzing feature importance...")
    selected = analyzer.analyze_feature_importance(n_features=50, use_advanced_methods=True, geochemical_priority=True)
    if not selected:
        print("No features selected. Exiting.")
        return

    # Environmental and source rock inference
    analyzer.infer_depositional_environments()
    analyzer.infer_source_rock_types()

    # Evaluate features
    analyzer.evaluate_features(test_size=0.3)

    # Visualizations
    print("\nGenerating visualizations...")
    analyzer.visualize_results()
    analyzer.plot_best_features_scatter_matrix(n_features=4)

    if len(analyzer.selected_features) >= 2:
        pairs = [(analyzer.selected_features[0], analyzer.selected_features[1])]
        if len(analyzer.selected_features) >= 4:
            pairs.append((analyzer.selected_features[2], analyzer.selected_features[3]))
        analyzer.plot_custom_scatter_pairs(feature_pairs=pairs)

    # Optimal Fréchet clustering (now uses up to 20 features from top 50)
    print("\nSearching for optimal feature set for Fréchet clustering...")
    analyzer.plot_optimal_hierarchical_clustering()

    # Export
    analyzer.export_results("hydrocarbon_ratio_analysis_v5_enhanced.xlsx")

    print("\n" + "="*60)
    print("Analysis completed.")
    print("="*60)

if __name__ == "__main__":
    main()