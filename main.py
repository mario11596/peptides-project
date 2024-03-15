import convert as cnv
import fileCheck as check_file
import filterFile as fil
import modelFilter as mdFilter
import modelWrapper as mdWrapper

if __name__ == '__main__':
    check_file.check_file()
    cnv.transform_to_smile()
    fil.filter_columns_file()
    fil.replacement_missing_value()
    fil.data_standardization()
    fil.unique_value()
    fil.feature_selection_kendall_model()
    fil.feature_selection_kendall_model_without_corr()

    mdFilter.train_model_catalytic()
    mdFilter.train_model_amp()

    mdWrapper.catalytic_function()
    mdWrapper.amp_function()
