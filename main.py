import convert as cnv
import fileCheck as check_file
import filterFile as fil

if __name__ == '__main__':
     check_file.check_file()
     cnv.transform_to_smile()
     fil.filter_columns_file()
     fil.data_standardization()
     fil.unique_value()


