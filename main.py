import convert as cnv
import fileCheck as check_file
import filterFile as fil
import model as md
import model2 as md2

if __name__ == '__main__':
     check_file.check_file()
     cnv.transform_to_smile()
     fil.filter_columns_file()
     fil.data_standardization()
     fil.unique_value()
     fil.feature_selection_kendall_model()
     md.train_model_catalytic()
     md.train_model_amp()


     #md2.train_model_catalytic()



