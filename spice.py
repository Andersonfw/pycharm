#  Copyright (c) 2023. Vieira Filho Tecnológia

"""
Created on março 20 17:35:30 2023
@author: vieirafilho
"""
# import PySpice.Logging.Logging as Logging
# logger = Logging.setup_logging()
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory
from PySpice.Unit import *

# Define a biblioteca de modelos de transistor do MOSFET
mosfet_library = SpiceLibrary('mos.lib')

# Cria um novo circuito
circuit = Circuit('CMOS Inverter Delay')

# Adiciona um VDD e um GND ao circuito
circuit.V('input', '+VDD', circuit.gnd, 5@u_V)
circuit.V('output', 'OUT', circuit.gnd, 0@u_V)

# Adiciona os transistores MOSFET do circuito
circuit.X('NMOS', 'mosfet_library:nmos', 'input', 'mid', 'mid', circuit.gnd)
circuit.X('PMOS', 'mosfet_library:pmos', 'mid', 'output', 'output', '+VDD')

# Define o valor de resistência e capacitância para gerar o delay
R = 1@u_kOhm
C = 1@u_pF

# Adiciona um capacitor e um resistor ao circuito para gerar o delay
circuit.R('R_delay', 'mid', circuit.gnd, R)
circuit.C('C_delay', 'mid', circuit.gnd, C)

# Executa a simulação do circuito
simulator = circuit.simulator()
waveform = simulator.transient(step_time=1@u_ns, end_time=10@u_ns)

# Plota o resultado da simulação
import matplotlib.pyplot as plt
plt.plot(waveform.time, waveform['OUT'])
plt.xlabel('Time (ns)')
plt.ylabel('Voltage (V)')
plt.show()
