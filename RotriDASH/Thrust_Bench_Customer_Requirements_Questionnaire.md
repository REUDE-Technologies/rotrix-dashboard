# ROTRIX Thrust Bench Requirement Form

**Prepared by:** REUDE Technologies Pvt. Ltd.  
**Purpose:** To collect the basic hardware requirements for a customized thrust bench setup.

Partial information is acceptable and can be refined during further technical discussions.

---

## 1. Organization Details

|Field|Response|
|---|---|
|Organization Name||
|Contact Person||
|Email / Phone||
|Requirement Date||
|Installation Location||
|Expected Timeline||

---

## 2. Test Objective and Use Case

**Primary purpose of the test system:**

- [ ] R&D
- [ ] Production
- [ ] Validation
- [ ] QC
- [ ] Certification
- [ ] Other:

**Product testing scope:**

- [ ] Motor
- [ ] Propeller
- [ ] Motor + propeller
- [ ] Battery
- [ ] ESC
- [ ] Full propulsion unit
- [ ] Other:

**Bench type:**

- ( ) Horizontal thrust bench
- ( ) Vertical thrust bench
- ( ) Portable thrust bench
- ( ) Fixed lab setup
- ( ) Need REUDE recommendation

**Typical and maximum motor sizes/classes:**  
**Expected daily/weekly testing volume:**

**Required test profiles:**

- [ ] Manual test
- [ ] Automated sweep
- [ ] Endurance test
- [ ] Custom profile
- [ ] Other:

**Duty cycles required:**

- [ ] Continuous run
- [ ] Stepped load
- [ ] Ramp
- [ ] Endurance cycle
- [ ] Transient testing
- [ ] Other:

---

## 3. Motor / Propeller Specifications

**Motor types:**

- [ ] BLDC
- [ ] Inrunner
- [ ] Outrunner
- [ ] EDF
- [ ] Other:

**Thrust operating range:**  
**Torque operating range:**  
**RPM operating range:**  
**Power operating range:**  
**Motor KV range:**  
**Propeller size range:**  
**Reference motor-propeller combinations, if any:**  
**Motor shaft, hub, and mounting interface requirements:**  
**CW/CCW testing requirement:** ( ) CW ( ) CCW ( ) Both  
**Quick-swap mounting required:** ( ) Yes ( ) No

---

## 4. Electrical and System Inputs

**Power source:**

- ( ) Battery
- ( ) PSU
- ( ) Hybrid
- ( ) Need REUDE recommendation

**Input voltage range:**  
**Maximum continuous current:**  
**Maximum peak current:**  
**Preferred or existing ESC/controller:**

---

## 5. Sensors Required

- [ ] Thrust load cell
- [ ] Torque sensor
- [ ] RPM sensor
- [ ] Voltage sensor
- [ ] Current sensor
- [ ] Temperature sensor
- [ ] Vibration sensor
- [ ] Airflow / wind speed sensor
- [ ] Other:
- [ ] Need REUDE recommendation

**Preferred sensor range/accuracy, if any:**

---

## 6. Controller / DAQ Board

**Preferred board or DAQ:**

- ( ) Arduino
- ( ) ESP32
- ( ) STM32
- ( ) Raspberry Pi
- ( ) NI DAQ
- ( ) Custom REUDE controller board
- ( ) Existing board:
- ( ) Need REUDE recommendation

**Expected data sampling rate:**

- ( ) 10 Hz
- ( ) 50 Hz
- ( ) 100 Hz
- ( ) 500 Hz
- ( ) 1 kHz
- ( ) Need REUDE recommendation

---

## 7. Data Communication

**Preferred communication method:**

- [ ] USB
- [ ] Ethernet
- [ ] Wi-Fi
- [ ] Bluetooth
- [ ] CAN
- [ ] UART / Serial
- [ ] SD card logging
- [ ] Other:

**Preferred cable/connector, if any:**  
**Approximate cable length required:**

---

## 8. Real-Time Data and Monitoring

**Live streaming into ROTRIX DAQ required:** ( ) Yes ( ) No

**Historical benchmarking and multi-test comparison required:** ( ) Yes ( ) No

**Real-time parameters to monitor:**

- [ ] Voltage
- [ ] Current
- [ ] RPM
- [ ] Thrust
- [ ] Torque
- [ ] Temperature
- [ ] Mechanical power
- [ ] Electrical power
- [ ] Motor efficiency
- [ ] Propeller efficiency
- [ ] Overall system efficiency
- [ ] Other:

---

## 9. Data Logging and Reporting

**Parameters currently logged by ROTRIX:**  
Voltage, current, RPM, thrust, torque, temperature (up to 6 sensors), mechanical power, electrical power, motor efficiency, propeller efficiency, and overall system efficiency.

**Required logging/export formats:**

- [ ] CSV
- [ ] JSON
- [ ] XLSX
- [ ] PDF report
- [ ] Other:

**File structure or naming convention, if any:**  
**Timestamp format:**  
**Units:** ( ) Metric ( ) Imperial

---

## 10. Data Analysis and Dashboard Requirements

**Key performance parameters to evaluate:**

- [ ] Thrust
- [ ] Torque
- [ ] RPM
- [ ] Current
- [ ] Voltage
- [ ] Temperature
- [ ] Motor efficiency
- [ ] Propeller efficiency
- [ ] Overall system efficiency
- [ ] Other:

**Required plots / performance curves:**

- [ ] Thrust vs RPM
- [ ] Thrust vs Power
- [ ] Current vs Thrust
- [ ] Voltage drop behavior
- [ ] Thermal trends
- [ ] Anomaly detection
- [ ] Pass/fail benchmarking
- [ ] Other X-Y combinations:

**Expected dashboard outputs:**  
**Required report format and automation level:**  
**Sample raw datasets / processed datasets / plots / reports available:** ( ) Yes ( ) No

---

## 11. Existing Thrust Bed / DAQ Details (If Applicable)

For ROTRIX multiple device integration using existing hardware already available, please share details of all available thrust beds.

**Existing thrust bed details:**

- Thrust bed model:
- Sensors used: manufacturer, model, output type, calibration method:
- Microcontroller / DAQ board:
- Data sampling frequency:
- Communication protocol and baud rate between controller and computer:
- Communication cable/interface between controller and computer:
- Structure/order of the final sensor data output sent from the controller:
- Parameters/variables used in the sensor calibration equations:
- Photos/videos/drawings of the setup available: ( ) Yes ( ) No
- Existing firmware source code available for review: ( ) Yes ( ) No

If firmware source code cannot be shared, please provide:

- Output data structure:
- Scaling logic:
- Calibration equations:

---

## 12. Support Requirement

- ( ) On-site support
- ( ) Remote support
- ( ) Both
- ( ) Not required

**Training required:** ( ) Yes ( ) No  
**Warranty / AMC expectation:**

---

## 13. Additional Notes

Please mention any other requirement, space limitation, safety need, or preferred hardware brand.

**Response:**

---

## 14. Approval

|Field|Response|
|---|---|
|Filled by||
|Approval to proceed for discussion/quotation|( ) Yes ( ) No|
