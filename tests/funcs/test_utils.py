import pytest

import src.utils.funcs as funcs

import warnings
warnings.filterwarnings('ignore')

@pytest.mark.skip()
class Test__get_data_from_files:
    def test_raise_error_if_no_training_file(self):
        with pytest.raises(FileNotFoundError):
            funcs.get_data_from_files(train_filepath="missing.csv")

    def test_raise_error_if_no_testing_file(self):
        with pytest.raises(FileNotFoundError):
            funcs.get_data_from_files(test_filepath="missing.csv")

@pytest.mark.skip()
def test_deco():
    funcs.test_run_with_randomforest()