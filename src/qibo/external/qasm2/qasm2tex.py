#!/usr/bin/python
#
# File:   qasm2tex.py
# Date:   22-Mar-04
# Author: I. Chuang <ichuang@mit.edu>
#
# Python program to convert qasm to latex (and optionally generate ps/epsf/pdf)
#
# Usage:   qasm2tex in.qasm
#
# Outputs: latex file (to stdout)
#
# Notes: qasm instructions are as follows.  Lines begining with '#'
# are comments.  All other lines should be of the form <b>op<b>args
# where <b> is whitespace, and op-args pairs are:
#
# qubit   name,initval
# cbit    name,initval
# measure qubit
# H       qubit
# X	  qubit
# Y	  qubit
# Z	  qubit
# S       qubit
# T       qubit
# nop	  qubit
# zero    qubit
# discard qubit
# slash   qubit
# dmeter  qubit
# cnot    ctrl,target
# c-z     ctrl,target
# c-x     ctrl,target
# toffoli ctrl1,ctrl2,target
# ZZ      b1,b2
# SS      b1,b2
# swap    b1,b2
# Utwo    b1,b2
# space   qubit
# def     opname,nctrl,texsym
# defbox  opname,nbits,nctrl,texsym
#
# Where:
#
# def     - define a custom controlled single-qubit operation, with
#           opname  = name of gate operation
#           nctrl   = number of control qubits
#           texsym  = latex symbol for the target qubit operation
# defbox  - define a custom muti-qubit-controlled multi-qubit operation, with
#           opname  = name of gate operation
#           nbits   = number of qubits it acts upon
#           nctrl   = number of control qubits
#           texsym  = latex symbol for the target qubit operation
# qubit   - define a qubit with a certain name (all qubits must be defined)
#           name    = name of the qubit, eg q0 or j2 etc
#           initval = initial value (optional), eg 0
# cbit    - define a cbit with a certain name (all cbits must be defined)
#           name    = name of the cbit, eg c0
#           initval = initial value (optional), eg 0
# H       - single qubit operator ("hadamard")
# X       - single qubit operator 
# Y       - single qubit operator 
# Z       - single qubit operator
# S       - single qubit operator
# T       - single qubit operator
# nop     - single qubit operator, just a wire
# space   - single qubit operator, just an empty space
# dmeter  - measure qubit, showing "D" style meter instead of rectangular box
# zero    - replaces qubit with |0> state
# discard - discard qubit (put "|" vertical bar on qubit wire)
# slash   - put slash on qubit wire
# measure - measurement of qubit, gives classical bit (double-wire) output
# cnot    - two-qubit CNOT
# c-z     - two-qubit controlled-Z gate
# c-x     - two-qubit controlled-X gate
# swap    - two-qubit swap operation 
# Utwo    - two-qubit operation U
# ZZ      - two-qubit controlled-Z gate, symmetric notation; two filled circles
# SS      - two-qubit gate, symmetric; open squares
# toffoli - three-qubit Toffoli gate
#
#-----------------------------------------------------------------------------
#
# Patched 02-Nov-04 by P. Oscar Boykin to allow arbitrarily large circuits
# (old version used to run out when chr() returned a non-alpha character)
# 
#-----------------------------------------------------------------------------
#
# $Log: qasm2tex.py,v $
# Revision 1.21  2004/03/25 15:36:59  ike
# special case for bullet target
# switched ZZ to using filled circles
# SS is now the two-qubit op with open squares
#
# Revision 1.20  2004/03/25 05:32:35  ike
# added comments for new gates
#
# Revision 1.19  2004/03/25 05:09:54  ike
# moved qubit labels to def's
# added ZZ, slash, discard, dmeter
#
# Revision 1.18  2004/03/24 20:49:03  ike
# more comments
#
# Revision 1.17  2004/03/24 20:47:08  ike
# comments for S,T
#
# Revision 1.16  2004/03/24 20:40:58  ike
# comments for swap
#
# Revision 1.15  2004/03/24 20:40:30  ike
# added swap gate
#
# Revision 1.14  2004/03/24 20:16:18  ike
# comments
#
# Revision 1.13  2004/03/24 20:15:27  ike
# multi-qubit controlled multi-qubit gates now work
# added space
#
# Revision 1.12  2004/03/24 19:24:30  ike
# muliqubit gate targets can now be in any order
# error checking is done for duplicate targets
#
# Revision 1.11  2004/03/24 18:04:17  ike
# added multi qubit gates
#
# Revision 1.10  2004/03/24 16:38:55  ike
# added zero, S,T,U
#
# Revision 1.9  2004/03/24 04:39:36  ike
# added copyright
#
# Revision 1.8  2004/03/24 03:22:43  ike
# added more comments
#
# Revision 1.7  2004/03/24 03:12:55  ike
# qubits can now have initial values
#
# Revision 1.6  2004/03/24 00:36:06  ike
# multiple controls on qubit now work
#
# Revision 1.5  2004/03/23 23:59:44  ike
# custom gate def's now work; see test4.qasm
#
# Revision 1.4  2004/03/23 23:42:35  ike
# new version with global gate definition table
#
# Revision 1.3  2004/03/23 23:13:36  ike
# working version, switches between single and double wires automatically
#
# Revision 1.2  2004/03/23 21:05:29  ike
# rcs log
#
#-----------------------------------------------------------------------------
#
# Copyright (c) 2004 Isaac L. Chuang <ichuang@mit.edu>
#
# This file, qasm2tex, is part of qasm2circ
#
# qasm2tex is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# qasm2tex is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with qasm2tex; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# -*-Python-*-

import re
import sys
import os
import fileinput
from struct import *
from string import *

#-----------------------------------------------------------------------------

def do_error(msg):	# global error handler
    sys.stderr.write('ERROR: ' + msg + '\n')
    sys.exit(-1)

#-----------------------------------------------------------------------------

def num2name(num):	# convert a number to a name
	if( num == 0 ):
	  return "";
	elif( num <= 26 ):
	  return chr(num+64)
	else:
	  return chr( (num % 26) + 64) + num2name(num/26)

#-----------------------------------------------------------------------------

class qgate:		# quantum gate class

    def __init__(self,op,args,linenum):

        self.name = op			# gate name
        self.args = args		# arguments to gate
        self.qubits = args.split(',')	# name of qubits we act upon
        self.timeseq = 0		# time sequence number
        self.id = 0			# gate ID number (unique)
        self.endtex = ''		# latex to output after xymatrix
        self.xy = {}			# gate xy ID table
        self.yloc = {}			# y-location of qubits we act upon
        self.wiretype = {}		# wire type for this gate/qubit
        self.linenum = linenum		# line number of input where gate used

        # do a quick syntax check to make sure number of operands is correct
        # and that the gate exists
        if not self.name in GateMasterDef:
            s = (self.linenum, self.name, self.args)
            do_error("[qgate] OOPS! line %d unknown gate op %s on %s" % s)

        # retrieve information about gate from master table
        (self.nbits, self.nctrl, self.texsym) = GateMasterDef[self.name]
            
        # check if the operand has the right number of bits
        if (len(self.qubits) != self.nbits): # right # bits?
            s = (self.linenum, self.name + " " + self.args)
            do_error("[qgate] OOPS! line %d wrong number of qubits in %s" % s)

        # check for duplicate operands
        x = self.qubits
        if ([ x.count(qb) for qb in x ].count(1) < len(x)):
            s = (self.linenum, self.name + " " + self.args)
            do_error("[qgate] OOPS! line %d duplicate bit operands in %s" % s)

    def set_bittype(self,qb,cbit):	# set qubit type (cbit/qbit)
        self.wiretype[qb] = cbit

    def make_id(self,qb2idx):		# make gate ID's, eg gAB
        for qb in self.qubits:
            self.xy[qb] = self.xyid(qb2idx[qb])
            self.yloc[qb] = qb2idx[qb]	# y (vertical) location of qubit

    def xid(self):			# return ID string for gate timestep
        return('g%s' % (num2name(self.timeseq)))

    def xyid(self,qubitnum):		# return ID string for gate/qubit
        return('%s%s%s' % (self.xid(),'x',num2name(qubitnum)))

    def latex(self):			# output latex/xypic/xyqcirc for gate

        def defid(k,op):		# latex def for given gate & qubit
            myid = self.xy[self.qubits[k]]
            wires = ['\w','\W']		# \w = single, \W = double wire
            mywire = wires[self.wiretype[self.qubits[k]]]
            return('\def\%s{%s%s\A{%s}}' % (myid,op,mywire,myid))

        def get_wiretype(qubits):	# figure out wire type for verticals
            # if any control is classical (double-wire) then all should be
            if(sum([ self.wiretype[x] for x in qubits])>0):
                wt = '='		# wire type = cbit
            else:
                wt = '-'		# wire type = qubit
            return(wt)

        def do_multiqubit(nbits,nctrl,u):	# multiple-qubit operation
            # first do target qubits (big box)
            s = []
            targets = self.qubits[nctrl:]
            ytab = [ self.yloc[qb] for qb in targets ]
            idx = ytab.index(min(ytab))	# find which qubit is first
            qb = targets[idx]		# handle first qubit specially

            ytop = min(ytab)		# remember y location & ID of top qubit
            xytop = self.xy[qb]
            ybot = max(ytab)		# and bottom
            xybot = self.xy[targets[ytab.index(ybot)]]

            myid = self.xy[qb]		# top qubit gets \gnqubit{u}{ddd...}
            dstr = 'd'*(nbits-nctrl-1)
            wires = ['\w','\W']		# \w = single, \W = double wire
            w = wires[self.wiretype[qb]]
            s.append(r'\def\%s{\gnqubit{%s}{%s}%s\A{%s}}'%(myid,u,dstr,w,myid))
            firstqb = qb
            for qb in targets:		# loop over target bits
                if (qb==firstqb):	# skip first qubit
                    continue
                myid = self.xy[qb]	# non-first bits get \gspace{u}
                w = wires[self.wiretype[qb]]
                s.append(r'\def\%s{\gspace{%s}%s\A{%s}}' % (myid,u,w,myid))
                
            # now do control qubits
            controls = self.qubits[:nctrl]
            for k in range(nctrl):	# loop over all control qubits
                s.append(defid(k,r'\b'))		# bullets on controls

            # create vertical wires
            # if any control is classical (double-wire) then all should be
            wt = get_wiretype(controls)
            for qb in controls: 	# loop over all ctrl qubits
                # endtex = latex commands which appear after xymatrix body
                # such as the vertical wires
                if self.yloc[qb] < ytop:
                    self.endtex += r'\ar@{%c}"%s";"%s"' %(wt,xytop,self.xy[qb])
                else:
                    self.endtex += r'\ar@{%c}"%s";"%s"' %(wt,xybot,self.xy[qb])

            # done with multi-qubit op
            return(join(s,'\n'))		# return with latex def's

        def ctrl_op(nctrl,u):		# controlled operation
            s = []
            for k in range(nctrl):	# loop over all control qubits
                s.append(defid(k,r'\b'))		# bullets on controls
            s.append(defid(nctrl,u))	# add target op 
            s = '\n'.join(s)

            # create vertical wires
            qbtarget = self.xy[self.qubits[-1]]	
            wt = get_wiretype(self.qubits[0:-1])
            for qb in self.qubits[0:-1]: # loop over all ctrl-target pairs
                # endtex = latex commands which appear after xymatrix body
                # such as the vertical wires
                self.endtex += r'\ar@{%c}"%s";"%s"' % (wt,qbtarget,self.xy[qb])

            return(s)

        def check_multi_qubit_gate_targets(nctrl):
            # gate targets (not controls) must be consecutive bits
            ytab = [self.yloc[qb] for qb in self.qubits[nctrl:]]
            ytab.sort()
            for k in range(len(ytab)-1):
                if (ytab[k+1]-ytab[k]!=1):
                    s = (self.linenum, self.name + " " + self.args)
                    do_error('[qgate] OOPS! line %d multi-qubit gate targets not consecutive %s' % s)

        def double_sym_gate(texsym):
            wt = get_wiretype(self.qubits)
            qb0 = self.xy[self.qubits[0]]
            qb1 = self.xy[self.qubits[1]]
            self.endtex += r'\ar@{%c}"%s";"%s"' % (wt,qb0,qb1)
            return(defid(0,texsym) + '\n' + defid(1,texsym))

        # main routine to generate latex
        (nbits, nctrl, texsym) = GateMasterDef[self.name]
        if(self.name=='zero'):		# special for zero: no wire
            myid = self.xy[self.qubits[0]]
            return('\def\%s{%s\A{%s}}' % (myid,texsym,myid))
        if(self.name=='space'):		# special for space: no wire
            myid = self.xy[self.qubits[0]]
            return('\def\%s{\A{%s}}' % (myid,myid))
        if(self.name=='ZZ'):		# special for ZZ gate
            return(double_sym_gate(texsym))
        if(self.name=='SS'):		# special for SS gate
            return(double_sym_gate(texsym))
        if(self.name=='swap'):		# special for swap gate
            return(double_sym_gate(texsym))
        if(nbits-nctrl>1):			# multi-qubit gate
            check_multi_qubit_gate_targets(nctrl)
            return(do_multiqubit(nbits,nctrl,texsym))
        if(nctrl==0):
            return(defid(0,texsym))		# single qubit op
        else: 
            return(ctrl_op(nctrl,texsym))	# controlled-single-qubit op

#-----------------------------------------------------------------------------

class qasm_parser:	# parser for qasm; inputs lines, returns
    			# tables of comments, names, and gates

    def __init__(self,fp):

        self.nametab = []	# table of bit names
        self.gatetab = []	# table of gates
        self.typetab = []	# table of bit types (0=qubit, 1=cbit)
        self.comments = ''	# string with comments from original qasm file

        linenum = 0		# line number counting, for error messages
        
        for line in fp:		# loop over input lines

            linenum += 1	# line number counter
            
            if(line[0]=='#'):
                self.comments += line 
                continue
            else:
                self.comments += "% " + line 	# optional - include all input

            # qubit spec - syntax: qubit name
            m = re.compile('\s+qubit\s+(\S+)').search(line)
            if(m):
                self.nametab.append(m.group(1))	# add name
                self.typetab.append(0)		# add as qubit
                # print "qubit: %s" % m.group(1)
                continue

            # cbit spec - syntax: cbit name
            m = re.compile('\s+cbit\s+(\S+)').search(line)
            if(m):
                self.nametab.append(m.group(1))	# add name
                self.typetab.append(1)		# add as cbit
                # print "cbit: %s" % m.group(1)
                continue

            # gate definition spec - syntax: def name,num-ctrl-qubits,texsym
            # this is for controlled single-qubit operations only
            m = re.compile("\s+def\s+(\S+),'(.*)'").search(line)
            if(m):
                (name,nctrl) = m.group(1).split(',')
                tex = m.group(2)
                if(tex=='bullet'):	      # special for bullet, no \op{}
                    texsym = r'\b'
                elif(tex.find(r'\dmeter')>=0):  # if meas, don't put in \op{}
                    texsym = tex
                else:
                    texsym = '\op{%s}' % tex
                nctrl = int(nctrl)
                if name in GateMasterDef:
                    print("[qasm_parser] oops! duplicate def for op %s" % line)
                else:
                    GateMasterDef[name] = (nctrl+1, nctrl, texsym)
                # print "definition: %s" % m.group(1)
                continue

            # box-gate definition spec - syntax: defbox name,nbits,nctrl,texsym
            # this is for multi-qubit controlled multi-qubit operations
            m = re.compile("\s+defbox\s+(\S+),'(.*)'").search(line)
            if(m):
                (name,nbits,nctrl) = m.group(1).split(',')
                texsym = m.group(2)
                nbits = int(nbits)
                nctrl = int(nctrl)
                if name in GateMasterDef:
                    print("[qasm_parser] oops! duplicate def for op %s" % line)
                else:
                    GateMasterDef[name] = (nbits, nctrl, texsym)
                # print "definition: %s" % m.group(1)
                continue

            # gate acting on qubits
            m = re.compile('\s+(\S+)\s+(\S+)').search(line)
            if(m):
                op = m.group(1)
                args = m.group(2)
                self.gatetab.append(qgate(op,args,linenum))

#-----------------------------------------------------------------------------

class qcircuit:		# quantum circuit class

    def __init__(self,bitnames,typetab):

        self.initval = {}	# qubit initial values
        self.is_cbit = {}	# flags to see if a bit is qubit or cbit
        self.setnames(bitnames,typetab)	# set names & types of qubits
        self.qbtab = {}		# initialize qubit table (assoc array)
				# each element in qbtab holds an array
                                # of IDs for gates acting on that qubit
        self.qb2idx = {}	# translate from name to index
        k = 1
        for name in self.qubitnames:	# create index for name->idx translate
            self.qbtab[name] = []	# array of gates on this qubit
            self.qb2idx[name] = k	# index for this qubit
            # print "%% [qcircuit] qubit %s (id=%d)" % (name,k)
            k += 1
        self.optab = []		# initialize table of gates
        self.circuit = []	# initialize table of circuit timesteps
        self.matrix = []	# initialize null circuit matrix

    def setnames(self,names,types):	# set bit names and types (+ initval)

        def do_name(n,type):		# set names & extract initial values
            tmp = n.split(',')			# check for initial value
            self.qubitnames.append(tmp[0])	# add to name list
            self.is_cbit[tmp[0]] = type		# 0 = qubit, 1 = cbit
            if(len(tmp)>1):
                self.initval[tmp[0]] = tmp[1]	# add initial value for qubit

        self.qubitnames = []
        for k in range(len(names)):		# loop over qubit names
            do_name(names[k],types[k])		# process name and type

    def add_op(self,gate):	# add gate to circuit

        self.optab.append(gate)		# put gate into table of gates
        gate.id = len(self.optab)-1	# give the gate a unique ID number
        # print "%% adding op %s(%s) IDs: %s" % (gate.name,gate.args,
        #                                       join(gate.xy.values(),','))
        
        for qb in gate.qubits:		# put gate on qubits it acts upon
            if not qb in self.qbtab:	# check for syntax error
                s = (qb,gate.linenum,gate.name + ' ' + gate.args)
                do_error('[qcircuit] No qubit %s in line %d: "%s"' % s)
            if(len(self.qbtab[qb])==0):	# if first gate, timestep = 1
                ts = 1
            else:			# otherwise, timestep = last+1
                ts = self.optab[self.qbtab[qb][-1]].timeseq+1
            self.qbtab[qb].append(gate.id)
            if(ts>gate.timeseq):	# set timeseq number for gate
                gate.timeseq = ts	# to be largest of its qubits

        gate.make_id(self.qb2idx)	# make gate ID's (do after timestep)

        if(gate.timeseq > len(self.circuit)):	# add new timestep if necessary
            self.circuit.append([])
        self.circuit[gate.timeseq-1].append(gate.id)	# add gate to circuit
        
    def output_sequence(self):	# output time-sequence of gates
        k = 1				# timestep counter
        for timestep in self.circuit:	# loop over timesteps
            print("%%  Time %02d:" % k)
            for g in timestep:		# loop over events in this timestep
                op = self.optab[g]
                print("%%    Gate %02d %s(%s)" % (op.id, op.name,op.args))
            k += 1
        print("")

    def output_matrix(self):	# output circuit matrix, of qubit vs timestep

        if(len(self.matrix)==0):	# make circuit matrix if not done
            self.make_matrix()

        k = 0
        print("% Qubit circuit matrix:\n%")
        for y in self.matrix:	# loop over qubits
            print('%% %s: %s' % (self.qubitnames[k],', '.join(y)))
            k += 1

    def make_matrix(self):	# make circuit matrix, of qubit vs timestep
        
        self.matrix = []
        ntime = len(self.circuit)+2	# total number of timsteps
        wires = ['n','N']		# single or double wire for qubit/cbit

        for qb in self.qubitnames:	# loop over qubits
            self.matrix.append([])	# start with empty row
            k = 1			# timestep counter
            cbit = self.is_cbit[qb]	# cbit=0 means qubit type (single wire)
            gidtab = self.qbtab[qb]	# table of gate IDs
            for gid in gidtab:		# loop over IDs for gates on qubit
                g = self.optab[gid]	# gate with that ID
                while(g.timeseq>k):	# output null ops until gate acts
                    self.matrix[-1].append('%s  ' % wires[cbit])
                    k += 1		# increment timestep  
                g.set_bittype(qb,cbit)	# set qubit type (cbit/qubit)
                self.matrix[-1].append(g.xy[qb])
                k += 1			# increment timestep
                if(g.texsym=='\meter'):	# if measurement gate then cbit=1
                    cbit = 1
                if(g.texsym.find('\dmeter')>=0): # alternative measurement gate
                    cbit = 1
                if(g.name=='measure'):	# if measurement gate then cbit=1
                    cbit = 1		# switch to double wire
                if(g.name=='zero'):	# if zero gate then cbit=0
                    cbit = 0		# switch to single wire
            while(k<ntime):		# fill in null ops until end of circuit
                k += 1			# unless last g was space or discard
                if((g.name!='space')&(g.name!='discard')):
                    self.matrix[-1].append('%s  ' % wires[cbit])

    def qb2label(self,qb):	# make latex format label for qubit name

        m = re.compile('([A-z]+)(\d+)').search(qb)
        if(m):			# make num subscript if name = alpha+numbers
            label = "%s_{%s}" % (m.group(1),m.group(2))
        else:
            label = qb			# othewise use just what was specified
        if(self.is_cbit[qb]):
            if qb in self.initval:	# qubit has initial value?
                label = r'   {%s = %s}' % (label,self.initval[qb])
            else:
                label = r'   {%s}' % (label)
        else:
            if qb in self.initval:	# qubit has initial value?
                label = r'\qv{%s}{%s}' % (label,self.initval[qb])
            else:
                label = r' \q{%s}' % (label)
        return(label)

    def output_latex(self, path):	# output latex with xypic for circuit

        fd = open(path, 'w')
        if(len(self.matrix)==0):	# make circuit matrix if not done
            self.make_matrix()

        fd.write("\n")
        fd.write("\documentclass[11pt]{article}\n")        # output latex header
        fd.write("\input{xyqcirc.tex}\n")

        # now go through all gates and output latex definitions
        fd.write("\n")
        fd.write("% definitions for the circuit elements\n")
        for g in self.optab:
            fd.write(g.latex())		# output \def\gXY{foo} lines

        # now output defs for qubit labels and initial states
        fd.write("\n")
        fd.write("% definitions for bit labels and initial states\n")
        for j in range(len(self.matrix)):
            qb = self.qubitnames[j]
            fd.write("\def\\b%s{%s}\n" % (num2name(j+1),self.qb2label(qb)))

        # now output circuit
        fd.write("\n")
        # print r'\xymatrix@R=15pt@C=12pt{'
        fd.write("% The quantum circuit as an xymatrix\n")
        fd.write("\\xymatrix@R=5pt@C=10pt{")

        ntime = len(self.circuit)+2	# total number of timsteps
        j = 0				# counter for timestep
        stab = []			# table of strings
        for y in self.matrix:		# loop over qubits
            qb = self.qubitnames[j]	# qubit name
            ops = ' &'.join(map(lambda x:'\\'+x,y))
            stab.append("\\b%s & %s" % (num2name(j+1),ops))
            j += 1			# increment timestep
        stab[0] = '    ' + stab[0]
        fd.write('\n\\\\  '.join(stab))

        # now go through all gates and output final latex (eg vertical lines)
        fd.write("%\n")
        fd.write("% Vertical lines and other post-xymatrix latex %\n")
        for g in self.optab:
            if(g.endtex!=""):
                fd.write(g.endtex)		# output end latex commands
                fd.write("\n")

        # now end the xymatrix & latex document
        fd.write("}\n")
        fd.write('\n')
        fd.write("\end{document}\n")
        fd.close()

#-----------------------------------------------------------------------------
# master gate definition table (global definition)
#
# Format = name : (nbits, nctrl, texsym)
#
# where:
#
# name     - text name of the gate op
# nbits    - total number of qubits gate acts upon
# nctrl    - number of control qubits
# texsym   - latex code for the operator target qubit
#
# This model assumes single qubit operations and multiple-qubit controlled
# single qubit operations.
#
# Note that GateMasterDef is modified by qasm_parser

GateMasterDef = {'cnot'     : ( 2 , 1 , '\o'        ),
                 'c-z'      : ( 2 , 1 , '\op{Z}'    ),
                 'c-x'      : ( 2 , 1 , '\op{X}'    ),
                 'measure'  : ( 1 , 0 , '\meter'    ),
                 'dmeter'   : ( 1 , 0 , '\dmeter{}' ),
                 'h'        : ( 1 , 0 , '\op{H}'    ),
                 'H'        : ( 1 , 0 , '\op{H}'    ),
                 'X'        : ( 1 , 0 , '\op{X}'    ),
                 'Y'        : ( 1 , 0 , '\op{Y}'    ),
                 'Z'        : ( 1 , 0 , '\op{Z}'    ),
                 'S'        : ( 1 , 0 , '\op{S}'    ),
                 'T'        : ( 1 , 0 , '\op{T}'    ),
                 'U'        : ( 1 , 0 , '\op{U}'    ),
                 'ZZ'       : ( 2 , 0 , r'\b'       ),
                 'SS'       : ( 2 , 0 , '\sq'       ),
                 'zero'     : ( 1 , 0 , '\z'        ),
                 'nop'      : ( 1 , 0 , '*-{}'      ),
                 'discard'  : ( 1 , 0 , '\discard'  ),
                 'slash'    : ( 1 , 0 , '\slash'    ),
                 'space'    : ( 1 , 0 , ''          ),
                 'swap'     : ( 2 , 0 , r'\t'       ),
                 'toffoli'  : ( 3 , 2 , r'\o'       ),
                 'Utwo'     : ( 2 , 0 , 'U'         ),
                 'RX'       : ( 1 , 0 , '\op{RX_\\theta}' ),
                 'RY'       : ( 1 , 0 , '\op{RY_\\theta}' ),
                 'RZ'       : ( 1 , 0 , '\op{RZ_\\theta}' )
                 }
                
#-----------------------------------------------------------------------------
# main program

def qasm2tex(path):
	#sys.stdout = open(path+'.tex', "w")
	qp = qasm_parser(fileinput.input(path+'.qasm'))	# parse the qasm file
	qc = qcircuit(qp.nametab,qp.typetab)	# initialize the circuit
	for g in qp.gatetab:			# add each gate to the circuit
    		qc.add_op(g)

	#print(qp.comments.replace('#','%'))	# output comments
	#qc.output_sequence()			# output time sequence of ops
	#qc.output_matrix()			# output matrix of qubit/timesteps
	qc.output_latex(path+'.tex')			# output latex code
	#sys.stdout = sys.__stdout__
