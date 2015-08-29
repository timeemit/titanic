require 'csv'
require 'awesome_print'
require 'colorize'

# Randomly split 891 records from prepare/pruned.csv into
# 535 records into train.csv
# 178 records into cross_validation.csv
# 178 records into test.csv

# Read
all_rows = CSV.read('./prepare/pruned.csv', headers: true, return_headers: true).to_a
headers = all_rows.shift

# Shuffle
all_rows.shuffle!

# Export Function
def matrix_to_csv_file(path, matrix)
  # Open File
  CSV.open(path, 'wb') do |csv|
    # Export Rows
    matrix.each do |array|
      csv << array
    end
  end
end

# Training Set
training_set = [ headers ] | all_rows.shift(535)
matrix_to_csv_file('./train/train.csv', training_set)

# Cross Validation Set
cross_validation_set = [ headers ] | all_rows.shift(178)
matrix_to_csv_file('./train/cross_validation.csv', cross_validation_set)

# Cross Validation Set
test = [ headers ] | all_rows.shift(178)
matrix_to_csv_file('./train/test.csv', test)

raise "Remaining Records!" unless all_rows.empty?
