$begin = 1
$end = $begin
$seed = $begin
$jump = 1

while(1)
{
python .\1_Model_Making.py $begin $end $seed
python .\2_FWMod.py $begin $end
python .\3a_JMI_target.py $begin $end
python .\3b_JMI_Prefocusing.py $begin $end
$begin += $jump
$end += $jump
$seed += $jump
}