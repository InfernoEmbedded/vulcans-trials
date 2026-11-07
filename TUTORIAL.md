# End-to-End Tutorial: Designing, Auto-Placing, and Auto-Routing a PCB with Vulcan's Trials

<!-- SPDX-License-Identifier: GPL-3.0-only -->

This tutorial walks through a complete PCB design workflow from an empty KiCad project to a routed, DRC-clean 2-layer board using **Vulcan's Trials**.

We will build the included sample project in [`examples/esphome_1wire_sensor/`](./examples/esphome_1wire_sensor/): an **ESP32-C3 ESPHome 1-Wire Temperature Sensor** node with a USB connector (`J1`), a `CH340N` USB-to-UART programming bridge (`U2`), a `3.3V` linear regulator (`U1`), an `ESP32-C3-WROOM-02` Wi-Fi/BLE module (`U3`), a status LED (`D1`), and a 3-pin 1-Wire `DS18B20` sensor header (`J2`).

---

## 1. Designing the Schematic & Grouping Related Components

Vulcan's Trials reads your `.kicad_sch` schematic alongside your `.kicad_pcb` board file to extract **functional placement clusters** and **decoupling capacitor affinity**. Structuring your schematic cleanly allows the auto-placer to keep functional sub-circuits tight and place decoupling capacitors right next to the IC power pins they protect.

### Rule 1: Functional Clusters via Bounding Boxes or Sub-Sheets
When `Board::load_schematic_clusters` parses the `.kicad_sch` file, it identifies functional clusters from:
1. **Bounding Boxes:** Graphical rectangles (`rectangle`) or closed dashed outlines (`polyline`) drawn around related groups of components in KiCad Schematic Editor. Any text placed inside the box automatically names the cluster (or it is auto-named after the primary IC/connector).
2. **Hierarchical Sub-Sheets:** Schematic sheets (`sheet`) representing sub-circuits (e.g. power supply, transceiver, sensors).
3. **Section Text Headers (Fallback):** In simple flat schematics without bounding boxes, text annotations whose first line starts with a digit and contains a period (`1. ...`, `2. ...`, `3. ...`) group components located in that region.

```mermaid
flowchart LR
    subgraph C1["1. USB & UART BRIDGE"]
        J1["J1: USB_B_Mini"] --> U2["U2: CH340N"]
        C3["C3: 100nF"] --- U2
    end
    subgraph C2["2. 3.3V LDO POWER"]
        C1["C1: 10uF"] --- U1["U1: AMS1117-3.3"]
        U1 --- C2["C2: 10uF"]
    end
    subgraph C3_MCU["3. ESP32-C3 MCU"]
        R1["R1: 10k EN"] --> U3["U3: ESP32-C3-WROOM-02"]
        U3 --- C4["C4: 10uF"]
        U3 --- C5["C5: 100nF"]
        U3 --> R2["R2: 1k"] --> D1["D1: GREEN LED"]
    end
    subgraph C4_SENS["4. 1-WIRE SENSOR"]
        R3["R3: 4.7k Pull-up"] --> J2["J2: DS18B20 1x03"]
        J2 --- C6["C6: 100nF"]
    end
    J1 -- "VBUS" --> U1
    U1 -- "+3.3V" --> U3
    U2 -- "UART_TX / UART_RX" --> U3
    U3 -- "ONEWIRE_DQ (GPIO4)" --> J2
```

In [`examples/esphome_1wire_sensor/esphome_1wire_sensor.kicad_sch`](./examples/esphome_1wire_sensor/esphome_1wire_sensor.kicad_sch), the schematic is divided into four functional quadrants, each headed by a bold text label at the top-left of its region:

1. **`1. USB & UART BRIDGE`** at `(20.32, 20.32)`:
   - `J1` (`Connector:USB_B_Mini`, footprint `Connector_USB:USB_Mini-B_Wuerth_65100516121_Horizontal`)
   - `U2` (`Interface_USB:CH340N`, footprint `Package_SO:JEITA_SOIC-8_3.9x4.9mm_P1.27mm`)
   - `C3` (`100nF` decoupling capacitor for `U2`)
2. **`2. 3.3V LDO POWER`** at `(139.70, 20.32)`:
   - `U1` (`Regulator_Linear:AMS1117-3.3`, footprint `Package_TO_SOT_SMD:SOT-223-3_TabPin2`)
   - `C1` (`10uF` input capacitor on `VBUS`) and `C2` (`10uF` output capacitor on `+3.3V`)
3. **`3. ESP32-C3 MCU`** at `(20.32, 109.22)`:
   - `U3` (`RF_Module:ESP32-C3-WROOM-02`, footprint `RF_Module:ESP32-C3-WROOM-02`)
   - `C4` (`10uF` bulk decoupling) and `C5` (`100nF` high-frequency decoupling)
   - `R1` (`10k` pull-up resistor on `ESP_EN`)
   - `R2` (`1k` current-limiting resistor) and `D1` (`LED_0603` status LED on `GPIO8`)
4. **`4. 1-WIRE SENSOR`** at `(139.70, 109.22)`:
   - `J2` (`Connector_Generic:Conn_01x03`, footprint `Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical_SMD_Pin1Left`)
   - `R3` (`4.7k` 1-Wire `ONEWIRE_DQ` pull-up resistor to `+3.3V`)
   - `C6` (`100nF` sensor supply bypass capacitor)

### Rule 2: Schematic Proximity for Decoupling Capacitors
Because all decoupling capacitors share the same global power nets (`+3.3V`, `VBUS`, `GND`), the PCB netlist alone cannot tell which `100nF` capacitor belongs to `U2` (`CH340N`) versus `U3` (`ESP32-C3`) or `J2` (`1-Wire Header`).

Vulcan's Trials solves this by comparing **schematic coordinates**:
- Every 2-pin capacitor (`C*`) connected between a power rail and `GND` is bound to the IC (`U*`) in the same schematic cluster whose schematic symbol is closest to it.
- During Simulated Annealing auto-placement, the cost function rewards placing that capacitor's power pad directly adjacent to the target IC's power pin (`decoupling_targets`).

> [!TIP]
> **Schematic Best Practice:** Always place each decoupling capacitor right next to its parent IC inside the same numbered section on the schematic sheet. Run KiCad's Electrical Rules Check (**Inspect → Electrical Rules Checker**) and resolve any unconnected pins or single-pin nets before moving to the PCB layout.

---

## 2. Configuring Netclasses in the Project

Before routing, define netclasses in **Board Setup → Design Rules → Net Classes** (stored in [`esphome_1wire_sensor.kicad_pro`](./examples/esphome_1wire_sensor/esphome_1wire_sensor.kicad_pro)) so Vulcan's Trials automatically sizes power rails and signal tracks:

| Netclass | Trace Width | Clearance | Via Size / Drill | Assigned Nets |
| :--- | :--- | :--- | :--- | :--- |
| **`Default`** | `0.25 mm` | `0.20 mm` | `0.60 mm` / `0.30 mm` | `UART_TX`, `UART_RX`, `ONEWIRE_DQ`, `ESP_EN`, `STATUS_LED`, `LED_ANODE` |
| **`Power`** | `0.50 mm` | `0.20 mm` | `0.60 mm` / `0.30 mm` | `VBUS`, `+3.3V`, `GND` |
| **`USB_Diff`** | `0.25 mm` | `0.15 mm` | `0.60 mm` / `0.30 mm` | `USB_D+`, `USB_D-` |

---

## 3. Laying Out the Board Outline & Locking Fixed Components

In PCB design, connectors, mounting holes, and RF modules with edge-mounted antennas have strict mechanical positions determined by the enclosure and RF keepout rules. Everything else (LDOs, bridge ICs, pull-up resistors, decoupling capacitors, indicator LEDs) can be placed automatically.

### Step 3.1: Draw the `Edge.Cuts` Outline
1. Open the PCB in KiCad PCB Editor (`pcbnew`) and run **Tools → Update PCB from Schematic (`F8`)** to import all 15 footprints.
2. Select the `Edge.Cuts` layer and draw a `48.0 mm × 32.0 mm` rectangular board outline from `(100.0, 100.0)` to `(148.0, 132.0)`.

### Step 3.2: Place and Lock Mechanically Constrained Components
Place the three edge-constrained components and **Lock** them (select the footprint → press `L`, or check **Locked** in Footprint Properties):

1. **`J1` (USB Connector — Left Edge):**
   - Place at `(103.5, 116.0)` with rotation `-90°` (`270°`) so the USB receptacle faces outward on the left board edge (`x = 100.0`).
   - Press `L` to **Lock** `J1`.
2. **`U3` (`ESP32-C3-WROOM-02` — Right Edge with Antenna Overhang):**
   - Place at `(140.0, 116.0)` with rotation `-90°` (`270°`) so the module's onboard PCB antenna overhangs past the right board edge (`x = 148.0`).
   - Add a **Rule Area / Keepout Zone** (`No Copper`, `No Tracks`, `No Vias`, `No Copper Fill` on `F.Cu` and `B.Cu`) covering the antenna region (`x = 146.5 .. 158.5`, `y = 101.5 .. 130.5`) so no ground pour or signal trace detunes the 2.4 GHz antenna.
   - Press `L` to **Lock** `U3`.
3. **`J2` (3-Pin 1-Wire Sensor Connector — Bottom Edge):**
   - Place at `(124.0, 128.0)` with rotation `90°` near the bottom edge of the PCB for external sensor wiring.
   - Press `L` to **Lock** `J2`.

### Step 3.3: Add Top & Bottom Ground Pour Zones
Add filled copper zones assigned to `GND` on both `F.Cu` and `B.Cu` covering the board outline `(100.0, 100.0)` to `(148.0, 132.0)`. Vulcan's Trials uses these zones to recognize plane nets and automatically inserts `GND` return and stitching vias after routing.

### Step 3.4: Leave Remaining Components Unlocked
Leave `U1` (`AMS1117-3.3`), `U2` (`CH340N`), `C1`–`C6`, `R1`–`R3`, and `D1` **unlocked** anywhere inside the board outline (or move them into Vulcan's Trials' **Holding Palette**).

---

## 4. Auto-Placing and Auto-Routing with Vulcan's Trials

You can run Vulcan's Trials either interactively from the GUI (or the KiCad ActionPlugin toolbar button) or headlessly from the command line.

### Option A: Interactive GUI Workflow
1. Launch Vulcan's Trials with the sample board:
   ```bash
   cargo run --release -- \
     --input examples/esphome_1wire_sensor/esphome_1wire_sensor.kicad_pcb \
     --output /tmp/esphome_1wire_sensor_routed.kicad_pcb
   ```
   *(If KiCad 10 is open with the board loaded, check **Enable Live Sync** in the **KiCad 10 IPC Live Sync** panel to stream placement and routing updates directly into `pcbnew`.)*
2. In the **Auto-Placer & Auto-Router Settings** panel on the left sidebar:
   - Ensure **Place Time (s)** is set to at least `120` seconds (the default is `120 s`). Running placement for at least 120 seconds gives the multi-threaded Parallel Tempering Simulated Annealing engine ample cooling cycles and replica swaps across temperatures to untangle ratsnest crossings, minimize total Manhattan and Euclidean wirelength, and cluster passives and decoupling capacitors tightly around their respective ICs before SAT courtyard legalization.
   - Hover over any button, slider, or checkbox in the sidebar to view detailed tooltips explaining each parameter's physical and algorithmic effect.
3. Click **`📐 Auto-Place Only`** (or **`⚡ Auto-Place & Route`**):
   - Locked components (`J1`, `U3`, `J2`) remain anchored at their exact mechanical coordinates.
   - Parallel Tempering Simulated Annealing pulls `U2` and `C3` toward `J1` (Cluster 1), places `U1` with input/output caps `C1` and `C2` between `J1` and `U3` (Cluster 2), arranges `C4`, `C5`, `R1`, `R2`, and `D1` around `U3` (Cluster 3), and pulls `R3` and `C6` next to `J2` (Cluster 4).
   - Courtyard legalization guarantees `0` courtyard overlaps while keeping all unlocked footprints outside the `U3` antenna keepout zone.
4. Once placement completes (or directly via **`⚡ Auto-Route Only`**), the PathFinder negotiated-congestion router:
   - Routes `VBUS` and `+3.3V` at `0.50 mm` width (`Power` netclass).
   - Routes `USB_D+` and `USB_D-` between `J1` and `U2`.
   - Routes `UART_TX`, `UART_RX`, `ESP_EN`, `ONEWIRE_DQ`, `STATUS_LED`, and `LED_ANODE` across `F.Cu` and `B.Cu`.
   - Inserts ground return vias and stitches the `F.Cu` and `B.Cu` `GND` planes.
5. Optional: Click **`🌿 Swap to Organic Tracks`** to smooth 45° segments into continuous tangent arcs, then click **`💾 Save Output (.kicad_pcb + .ses)`**.

### Option B: Headless CLI Workflow
To run both auto-placement and auto-routing from a terminal or CI script, pass `--duration 120` (or higher) so the auto-placer runs for at least 120 seconds to achieve optimal annealing convergence:

```bash
./target/release/vulcans-trials \
  --headless \
  --mode both \
  --duration 120 \
  --passes 10 \
  --input examples/esphome_1wire_sensor/esphome_1wire_sensor.kicad_pcb \
  --output /tmp/esphome_1wire_sensor_routed.kicad_pcb
```

If you used `--lock` on the command line instead of locking footprints inside KiCad, you can also pass explicit locked references:

```bash
./target/release/vulcans-trials \
  --headless \
  --mode both \
  --duration 120 \
  --lock J1,U3,J2 \
  --input examples/esphome_1wire_sensor/esphome_1wire_sensor.kicad_pcb \
  --output /tmp/esphome_1wire_sensor_routed.kicad_pcb
```

---

## 5. Verifying DRC & Flashing ESPHome

1. Run KiCad's CLI Design Rules Check (DRC) on the routed output board:
   ```bash
   kicad-cli pcb drc \
     --output /tmp/esphome_drc.json \
     --format json \
     /tmp/esphome_1wire_sensor_routed.kicad_pcb
   ```
   Verify `0` unconnected items and `0` copper clearance/short violations.
2. Open the routed board in KiCad, press **`B`** to refill the `F.Cu` and `B.Cu` `GND` copper zones around the newly routed tracks and stitching vias, and export Gerbers for fabrication.
3. Flash the board over USB using the included ESPHome configuration ([`examples/esphome_1wire_sensor/esphome_1wire_sensor.yaml`](./examples/esphome_1wire_sensor/esphome_1wire_sensor.yaml)):
   ```bash
   esphome run examples/esphome_1wire_sensor/esphome_1wire_sensor.yaml
   ```
