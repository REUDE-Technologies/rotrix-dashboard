# Femto Bolt + NVIDIA Jetson Orin Nano: Prerequisites and Setup Guide

This document summarizes **hardware and software prerequisites**, a **practical bring-up checklist**, and **curated references** (vendor docs, SDK sources, and background reading) for combining the **Orbbec Femto Bolt** time-of-flight / RGB-D camera with a **Jetson Orin Nano** developer kit.

**Product page (Femto Bolt):** [https://www.orbbec.com/products/tof-camera/femto-bolt/](https://www.orbbec.com/products/tof-camera/femto-bolt/)

**Jetson Orin Nano (NVIDIA):** [https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit](https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit)

---

## 1. What this combo is

- **Femto Bolt:** USB 3.2 host-connected depth + RGB camera (iToF depth pipeline; modes aligned with **Azure Kinect DK** style usage). Streams depth, IR, color, and includes an **IMU**; optional **sync** for multi-camera rigs.
- **Jetson Orin Nano:** ARM64 edge computer (CUDA-capable) for perception, SLAM, tracking, or ML on RGB/depth streams.

**Important:** Depth is computed on the **host** (Jetson), not inside the camera module. USB bandwidth, CPU load, and your application’s concurrent GPU work all affect stable frame rates.

---

## 2. Hardware prerequisites

### 2.1 Femto Bolt (from vendor product summary)

- **Interface:** USB **3.2 Gen 1** Type-C (5 Gbit/s class SuperSpeed USB).
- **Power:** USB-C bus power **or** optional **12 V DC, 2 A** barrel input (use vendor-approved supply and polarity).
- **RGB:** up to **4K** color (higher RGB resolution increases USB load; tune modes for your pipeline).
- **Depth:** iToF; **NFOV / WFOV** style depth modes (Kinect-class ecosystem alignment).
- **IMU:** 6DoF inertial data available to applications via SDK.
- **Multi-device:** **sync** connector / triggering options for aligned captures (see Orbbec hardware user guide for your revision).

### 2.2 Jetson Orin Nano developer kit

- **Storage:** High-endurance **microSD** (official images) or **NVMe** (if your carrier supports it and you use supported install paths).
- **Power:** Use the **NVIDIA-recommended** power supply for your kit revision; enable the appropriate **power mode** (`nvpmodel` / `jtop`) when running sustained perception + USB cameras.
- **Cooling:** Passive or active cooling per NVIDIA guidance; ToF + ML workloads are thermally sensitive (throttling affects real-time stability).
- **Peripherals:** USB keyboard, mouse, and display (or headless SSH) for first boot and debugging.

### 2.3 USB3 link quality (critical for Femto Bolt)

- **Short, high-quality USB 3.x cable** (preferably certified, avoid charge-only cables).
- Prefer **direct** connection to a **USB 3.x** port on the Jetson carrier when possible. If you use a hub, it must be a **powered USB 3.x hub** with adequate upstream bandwidth and power budget.
- Avoid USB2 extension cords or marginal cables; depth + color + IMU at higher modes will **drop** or **stall** when the link degrades to USB2.

### 2.4 Environment (ToF-specific)

- **Bright sunlight / strong IR** on the sensor can degrade ToF quality.
- **Reflective / transparent** materials cause missing depth or multipath-like artifacts.
- For metrology-grade results, plan for **calibration** and **temperature** stability.

---

## 3. Software prerequisites (Jetson)

### 3.1 JetPack / L4T baseline

1. Flash or SD-image the Jetson with a **JetPack** release appropriate to your project (match versions expected by any prebuilt binaries you use, e.g. specific `libc` / OpenCV / TensorRT stacks).
2. Complete **first-boot** setup (locale, user, network).
3. Apply **system updates** cautiously on embedded systems (test after updates; keep a known-good SD image backup).

**NVIDIA getting started (Orin Nano):** [https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide/getting_started](https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide/getting_started)

**Note on JetPack 6.x:** NVIDIA documentation states that **Jetson Orin Nano firmware must be updated before installing JetPack 6.x**. Follow the user guide for your kit revision.

### 3.2 Build tools (if compiling from source)

- `build-essential`, CMake, Git, pkg-config.
- For OpenCV / PCL / ROS2 builds, install **matching** dev packages or use containers.

### 3.3 Orbbec SDK on Jetson (ARM64)

Orbbec publishes **Linux arm64** SDK packages (`.deb` / release zip) suitable for Jetson-class boards; Femto Bolt is maintained in **OrbbecSDK v1** and **v2** (v2 recommended for new designs).

- **OrbbecSDK v2 (GitHub):** [https://github.com/orbbec/OrbbecSDK_v2](https://github.com/orbbec/OrbbecSDK_v2)
- **OrbbecSDK v1 (GitHub, releases with arm64 assets):** [https://github.com/orbbec/OrbbecSDK/releases](https://github.com/orbbec/OrbbecSDK/releases)

**Vendor system requirements (general SDK host expectations, often x64-focused in prose):** [https://doc.orbbec.com/OrbbecSDK/english/Environment_Configuration/Environment_Configuration.html](https://doc.orbbec.com/OrbbecSDK/english/Environment_Configuration/Environment_Configuration.html)

Always verify on the **release notes** for your exact SDK version:

- **arm64 / aarch64** package availability.
- **Kernel / Ubuntu** compatibility with your JetPack’s Ubuntu base.
- Known issues for **Jetson** (historical release notes mention stream-toggle / V4L2 fixes for Femto Bolt on Jetson Nano family; treat as a signal to test stream start/stop cycles on your Orin Nano build).

### 3.4 udev and permissions

Install Orbbec’s **udev rules** so non-root users can open the device. After installing rules:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Re-plug the camera or reboot once.

### 3.5 Optional: Azure Kinect–style API path

Femto Bolt is positioned for **Kinect-for-Azure–like** workflows. If you migrate AKDK applications, use Orbbec’s **K4A wrapper** / compatibility layer per current Orbbec documentation (check the doc tree version that matches your SDK).

**Development guide (entry point):** [https://doc.orbbec.com/OrbbecSDK/english/DevelopmentGuide/DevelopmentGuide.html](https://doc.orbbec.com/OrbbecSDK/english/DevelopmentGuide/DevelopmentGuide.html)

### 3.6 Optional stacks

- **ROS 2** wrapper packages (community or vendor; verify maintenance status and SDK version alignment).
- **Docker**: viable for reproducible dev, but USB device passthrough and real-time performance need careful configuration.

---

## 4. Bring-up checklist (recommended order)

1. **Image Jetson** with target JetPack; complete first boot; set static IP or reliable Wi-Fi/Ethernet.
2. **Cooling + power mode** appropriate for sustained load.
3. **Update firmware** if moving to JetPack 6.x (per NVIDIA user guide).
4. Install **Orbbec SDK** arm64 package **or** build from source per repo instructions.
5. Install **udev** rules; confirm device node permissions.
6. Connect Femto Bolt to a **verified USB3** port; run `lsusb` and confirm the device enumerates as a SuperSpeed device (troubleshoot cable/hub if not).
7. Launch **Orbbec Viewer** (or sample apps) and verify:
   - Depth + IR stable at your target resolution / FPS.
   - Color stream if needed (watch USB bandwidth when pushing 4K + depth).
   - IMU stream if your app depends on it.
8. Integrate application stack (OpenCV, ROS 2, TensorRT, etc.) and profile **CPU + GPU + USB** together.

**Known issues / troubleshooting (Orbbec doc hub):** [https://doc.orbbec.com/OrbbecSDK/english/7_Troubleshooting.html](https://doc.orbbec.com/OrbbecSDK/english/7_Troubleshooting.html)

---

## 5. Verification commands (quick)

```bash
# USB enumeration
lsusb

# Jetson clocks / power mode (example; see NVIDIA docs for your JP version)
sudo nvpmodel -q
```

Use Orbbec’s viewer logs for **frame drops**, **CRC**, or **USB** warnings when debugging instability.

---

## 6. Risk notes (planning)

- **SDK vs JetPack drift:** Prebuilt `.deb` packages may target specific Ubuntu/glibc versions; mismatches require building from source or using a supported container.
- **OpenGL / GUI samples:** Some SDK samples assume a desktop GL stack; headless or minimal images may need different sample entry points.
- **Real-time robotics:** Prefer **deterministic** USB topology (no flaky hub), isolate IRQ/USB load where possible, and cap camera modes before adding heavy DL.

---

## 7. References for quick knowledge

### 7.1 Vendor and SDK (primary)

- **Femto Bolt product page:** [https://www.orbbec.com/products/tof-camera/femto-bolt/](https://www.orbbec.com/products/tof-camera/femto-bolt/)
- **Orbbec SDK documentation hub:** [https://doc.orbbec.com/OrbbecSDK/english/index.html](https://doc.orbbec.com/OrbbecSDK/english/index.html)
- **Environment / system configuration:** [https://doc.orbbec.com/OrbbecSDK/english/Environment_Configuration/Environment_Configuration.html](https://doc.orbbec.com/OrbbecSDK/english/Environment_Configuration/Environment_Configuration.html)
- **Development guide:** [https://doc.orbbec.com/OrbbecSDK/english/DevelopmentGuide/DevelopmentGuide.html](https://doc.orbbec.com/OrbbecSDK/english/DevelopmentGuide/DevelopmentGuide.html)
- **Troubleshooting:** [https://doc.orbbec.com/OrbbecSDK/english/7_Troubleshooting.html](https://doc.orbbec.com/OrbbecSDK/english/7_Troubleshooting.html)
- **OrbbecSDK v2 (GitHub):** [https://github.com/orbbec/OrbbecSDK_v2](https://github.com/orbbec/OrbbecSDK_v2)
- **OrbbecSDK v1 releases (arm64 assets):** [https://github.com/orbbec/OrbbecSDK/releases](https://github.com/orbbec/OrbbecSDK/releases)

### 7.2 NVIDIA Jetson Orin Nano

- **Jetson Orin Nano developer kit:** [https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit](https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit)
- **Getting started (user guide):** [https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide/getting_started](https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide/getting_started)

### 7.3 Azure Kinect ecosystem (API / mode alignment context)

Femto Bolt is often used where **Azure Kinect DK** APIs or modes are assumed. Microsoft’s depth + sensor documentation remains a practical reference for **NFOV/WFOV**, **depth modes**, and **sensor synchronization** concepts:

- **Azure Kinect DK documentation (Microsoft Learn):** [https://learn.microsoft.com/en-us/azure/kinect-dk/](https://learn.microsoft.com/en-us/azure/kinect-dk/)

### 7.4 Papers and surveys (ToF / depth cameras)

These are useful for **fundamentals** (CW modulation / phase iToF), **calibration**, and **failure modes** (multipath, motion blur, ambient light)—not vendor-specific tuning.

1. **Horaud, R., Hansard, M., Evangelidis, G., Ménier, C.** “An Overview of Depth Cameras and Range Scanners Based on Time-of-Flight Technologies.” *arXiv preprint* arXiv:2012.06772 (2020).  
   [https://arxiv.org/abs/2012.06772](https://arxiv.org/abs/2012.06772) (PDF: [https://arxiv.org/pdf/2012.06772](https://arxiv.org/pdf/2012.06772))

2. **Foix, S., Alenyà, G., Torras, C.** “Lock-in Time-of-Flight (ToF) Cameras: A Survey.” *IEEE Sensors Journal*, 2011 (widely cited survey on ToF camera principles and applications).  
   IEEE Xplore: search title “Lock-in Time-of-Flight (ToF) Cameras: A Survey”.

3. **Rapp, H. et al.** (and related indirect-ToF sensor literature) — for **indirect ToF** sensor technology overviews, see IEEE magazine surveys on **indirect time-of-flight** (search: “Review of Indirect Time-of-Flight Technologies” on IEEE Xplore).

4. **Kahlmann, T., Remondino, F., Ingensand, H.** “Calibration for increased accuracy of the range imaging camera SwissRanger.” *ISPRS* imaging and calibration themes (useful calibration framing; sensor-specific but pedagogical).  
   (Search title on [https://www.isprs.org/](https://www.isprs.org/) or scholar indexes.)

5. **Multipath / error models in ToF** — search keywords: *“ToF multipath interference”*, *“indirect ToF phase unwrapping”* for articles matching your accuracy targets.

---

## 8. Document control

- **Scope:** Prerequisites + bring-up checklist + references (not a substitute for revision-specific Orbbec / NVIDIA PDFs).
- **Maintainer note:** Re-verify **arm64 package** and **JetPack** pairing on the day of install; embedded stacks change frequently.

---

*End of guide.*
