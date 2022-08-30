run_rapl() {
# run with RAPL
  sudo mkdir -p AutomationOutputs/rapl_"$1"
  echo "starting powerstat" &
  # start up powerstat, redirect output to txt file, prevent output from showing in terminal
  sudo powerstat -DRgf -d=0 1 500 | sudo tee AutomationOutputs/rapl_"$1"/rapl_output_"$1".txt > /dev/null &
  echo "running script with RAPL coverage" &&
  echo "waiting for script to finish" &&
  sudo -E PATH="$PATH" python3 dsc.py --sample="${2:-None}" &&
  echo "script finished, killing processes..."
  sudo pkill -f powerstat ;
  sudo pkill -f dsc.py ;
  echo "all processes killed"
  return 0
}

run_ct() {
# run with carbontracker
echo "running script with CarbonTracker coverage" &&
sudo -E PATH="$PATH" python3 dsc.py --use_ct --suffix="ct_$1" --sample="${2:-None}" &&
return 0
}

echo "installing powerstat"
sudo apt install powerstat
echo "installing dependencies"
pip3 install -r requirements.txt
sudo lshw -xml | sudo tee lshw.xml > /dev/null  # print out hardware info
sudo mkdir AutomationOutputs
sudo mkdir Plots
sudo mkdir Model_Info

for i in 1
do
  # run with RAPL
  run_rapl $i "$1"
  # run with carbontracker
  run_ct $i "$1"
done
