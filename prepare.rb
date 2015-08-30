require 'csv'
require 'awesome_print'
require 'colorize'

# Copy over passenger_id
# Copy over pclass
# Copy over age
# Copy over survived
# Copy over sibsp
# Copy over parch
# Copy over fare
# Parse Name into Last Name and First Name Fields
# Convert Male / Female Sex to 1 / 0 Gender, respectively
# Parse Cabin Classes "A", "B", "C"... into 1, 2, 3... leaving 0 for unlabeled
# Convert Embarked C / S / Q to 1 / 2 / 3, respectively
# Number of (14 years elder | 14 years younger ) fellow travelers on the same ticket
# Number of fellow travelers on the same ticket within 14 years of age.
# Percentage of travelers on the same ticket with same last name

DATA_FILE = './kaggle-data/test.csv'
PRUNED_FILE = './predict/input.csv'

RAW_ATTRIBUTES = [
  :passenger_id,
  # :survived,
  :pclass,        # [ 1, 2, 3 ]
  :name,
  :sex,           # %w( male female )
  :age,
  :sibsp,
  :parch,
  :ticket,        # No. identifying group
  :fare,          # $$$$
  :cabin,         # /\w[0-9]+/
  :embarked       # %w( C S Q )
]

PRUNED_ATTRIBUTES = [
  :passenger_id,  # same
  # :survived,      # same
  :pclass,        # same
  :age,           # same
  :sibsp,         # same
  :parch,         # same
  :fare,          # same
  :first_name,    # Name Before ","
  :last_name,     # Name After ". "
  :gender,        
  :cabin_class, 
  :embarkation_index, 
  :elder_travelers,
  :younger_travelers,
  :peer_travelers,
  :familial_travelers
]

COMMON_ATTRIBUTES = RAW_ATTRIBUTES & PRUNED_ATTRIBUTES

FIRST_NAME_REGEX = /\.\s\(*\w+/ # /(?<=(Major)|(Mr)|(Sir)|(Mrs)|(Master)|(Don)|(Dr)|(Mme)|(Lady)|(Ms)|(Rev)|(Miss))\.\s\(*/
LAST_NAME_REGEX = /\w+(?=,)/

GENDER_CONVERTER = {
  'male' => 1,
  'female' => 2
}

CABIN_REGEX = /^\w/
CABIN_CONVERTER = { 
  nil => 0,
  '' => 0,
  'A' => 1,
  'B' => 2,
  'C' => 3,
  'D' => 4,
  'E' => 5,
  'F' => 6,
  'G' => 7,
  'T' => 20
}

EMBARKATION_CONVERTER = { 
  nil => 0,
  "C" => 1, 
  "S" => 2, 
  "Q" => 3 
}

Struct.new('RawPassenger', *RAW_ATTRIBUTES)
Struct.new('PrunedPassenger', *PRUNED_ATTRIBUTES)

# Construct Ticket Hashes
ticket_groups = {}
extract_ticket_groups = Proc.new do |row|
  ticket = row['Ticket']
  age_surname = {
    # survived: row['Survived'],
    sex: row['Sex'],
    age: row['Age'].to_i,
    surname: row['Name'].match(LAST_NAME_REGEX).to_s 
  }
  if ticket_groups.has_key? ticket
    ticket_groups[ticket] << age_surname
  else
    ticket_groups[ticket] = [ age_surname ]
  end
end

# Parse the Training Data for Group Information
CSV.foreach(DATA_FILE, headers: :first_row, &extract_ticket_groups)

# ap "#{ticket_groups.values.select{|array| array.size == 1}.length} are empty"
# ap ticket_groups.values.reject{|array| array.size == 1}

CSV.open(PRUNED_FILE, 'wb') do |output_row|
  # Header
  output_row << PRUNED_ATTRIBUTES.map(&:to_s)

  CSV.foreach(DATA_FILE, headers: :first_row) do |input_row|
    raw_passenger = Struct::RawPassenger.new(*input_row.fields)
    parsed_passenger = Struct::PrunedPassenger.new

    # Copy over common attributes
    COMMON_ATTRIBUTES.each do |common_attribute|
      parsed_passenger[common_attribute] = raw_passenger[common_attribute]
    end

    parsed_passenger.age ||= -1 # Unknown

    # First Name just follows the period, a space, and maybe a open parens
    parsed_passenger.first_name = raw_passenger.name.match(FIRST_NAME_REGEX).to_s.gsub(/[\.\s\(]/, '')
    raise "Could Not Parse First Name: #{raw_passenger.name}" if parsed_passenger.first_name.empty?

    # Last Name is everything before a comma
    parsed_passenger.last_name = raw_passenger.name.match(LAST_NAME_REGEX).to_s
    raise "Could Not Parse Last Name: #{raw_passenger.name}" if parsed_passenger.last_name.empty?

    parsed_passenger.gender = GENDER_CONVERTER[raw_passenger.sex]
    raise "Could Not Determine Gender: #{raw_passenger.sex}" if ['', nil].include? parsed_passenger.gender

    parsed_passenger.cabin_class = CABIN_CONVERTER[ raw_passenger.cabin.to_s.match(CABIN_REGEX).to_s ]
    raise "Could Not Determine Embarkation: #{raw_passenger.cabin}" if parsed_passenger.cabin_class.nil?

    parsed_passenger.embarkation_index = EMBARKATION_CONVERTER[raw_passenger.embarked]
    raise "Could Not Determine Embarkation: #{raw_passenger.embarked}" unless [0, 1, 2, 3].include? parsed_passenger.embarkation_index

    # Group
    group = ticket_groups[raw_passenger.ticket]

    if raw_passenger.age
      parsed_passenger.elder_travelers = group.select{|p| p[:age] && p[:age] - 14 >= raw_passenger.age.to_i}.size
      parsed_passenger.younger_travelers = group.select{|p| p[:age] && p[:age] + 14 <= raw_passenger.age.to_i}.size
      parsed_passenger.peer_travelers = group.select do |p| 
        p[:age] && ( p[:age] - 14 >= raw_passenger.age.to_i ) && ( p[:age] + 14 <= raw_passenger.age.to_i )
      end.size
    else
      parsed_passenger.elder_travelers = 0 
      parsed_passenger.younger_travelers = 0 
      parsed_passenger.peer_travelers = 0 
    end
    raise "Could Not Determine Elder Travelers: #{group}" if parsed_passenger.elder_travelers.nil?
    raise "Could Not Determine Younger Travelers: #{group}" if parsed_passenger.younger_travelers.nil?
    raise "Could Not Determine Peer Travelers: #{group}" if parsed_passenger.peer_travelers.nil?

    parsed_passenger.familial_travelers = group.select{|p| p[:surname] == parsed_passenger.last_name}.size
    raise "Could Not Determine Familial Travelers: #{group}" if parsed_passenger.familial_travelers.nil?

    output_row << parsed_passenger.to_a
  end
end

puts '**************'.green.bold
print '*** '.green.bold 
print 'Parsed'.green
puts ' ***'.green.bold
puts '**************'.green.bold
