import numpy as np
from .InstrumentController import InstrumentController
from qcodes.instrument_drivers.AlazarTech import ATS
from .ATS9371 import AlazarTech_ATS9371

class AcquisitionController(ATS.AcquisitionController):
    def __init__(self, name="alz_cont", alazar_name="Alazar1", **kwargs):
        self.alazar = AlazarTech_ATS9371(alazar_name)
        trigger = 1
        input_range_volts = 2.5
        trigger_level_code = int(128 + 127 * trigger / input_range_volts)
        with self.alazar.syncing():
            self.alazar.clock_source("EXTERNAL_CLOCK_10MHz_REF")
            #self.alazar.clock_source("INTERNAL_CLOCK")
            self.alazar.external_sample_rate(1_000_000_000)
            #self.alazar.sample_rate(1_000_000_000)
            self.alazar.clock_edge("CLOCK_EDGE_RISING")
            self.alazar.decimation(1)
            self.alazar.coupling1('DC')
            self.alazar.coupling2('DC')
            self.alazar.channel_range1(.02)
            #self.alazar.channel_range2(.4)
            self.alazar.channel_range2(.02)
            self.alazar.impedance1(50)
            self.alazar.impedance2(50)
            self.alazar.bwlimit1("DISABLED")
            self.alazar.bwlimit2("DISABLED")
            self.alazar.trigger_operation('TRIG_ENGINE_OP_J')
            self.alazar.trigger_engine1('TRIG_ENGINE_J')
            self.alazar.trigger_source1('EXTERNAL')
            self.alazar.trigger_slope1('TRIG_SLOPE_POSITIVE')
            self.alazar.trigger_level1(trigger_level_code)
            self.alazar.trigger_engine2('TRIG_ENGINE_K')
            self.alazar.trigger_source2('DISABLE')
            self.alazar.trigger_slope2('TRIG_SLOPE_POSITIVE')
            self.alazar.trigger_level2(128)
            self.alazar.external_trigger_coupling('DC')
            self.alazar.external_trigger_range('ETR_2V5')
            self.alazar.trigger_delay(0)
            #self.aux_io_mode('NONE') # AUX_IN_TRIGGER_ENABLE for seq mode on
            #self.aux_io_param('NONE') # TRIG_SLOPE_POSITIVE for seq mode on
            self.alazar.timeout_ticks(0)
        self.ic = InstrumentController()
        self.acquisitionkwargs = {}
        self.samples_per_record = None
        self.records_per_buffer = None
        self.buffers_per_acquisition = None
        # TODO(damazter) (S) this is not very general:
        self.number_of_channels = 2
        self.buffer = None
        self.time_array = None
        # make a call to the parent class and by extension, create the parameter
        # structure of this class
        super().__init__(name, alazar_name, **kwargs)
        self.add_parameter("acquisition", get_cmd=self.do_acquisition)

    def update_acquisitionkwargs(self, **kwargs):
        """
        This method must be used to update the kwargs used for the acquisition
        with the alazar_driver.acquire
        :param kwargs:
        :return:
        """
        self.acquisitionkwargs.update(**kwargs)

    def do_acquisition(self):
        """
        this method performs an acquisition, which is the get_cmd for the
        acquisiion parameter of this instrument
        :return:
        """
        value = self._get_alazar().acquire(acquisition_controller=self,
                                           **self.acquisitionkwargs)
        return value

    def pre_start_capture(self):
        """
        See AcquisitionController
        :return:
        """
        alazar = self.alazar
        self.samples_per_record = alazar.samples_per_record.get()
        self.records_per_buffer = alazar.records_per_buffer.get()
        self.buffers_per_acquisition = alazar.buffers_per_acquisition.get()
        sample_speed = alazar.get_sample_rate()
        t_final = self.samples_per_record / sample_speed
        self.time_array = np.arange(0, t_final, 1 / sample_speed)
        self.buffer = np.zeros(self.samples_per_record *
                               self.records_per_buffer *
                               self.number_of_channels)


    def pre_acquire(self):
        """
        See AcquisitionController
        :return:
        """
        # this could be used to start an Arbitrary Waveform Generator, etc...
        # using this method ensures that the contents are executed AFTER the
        # Alazar card starts listening for a trigger pulse
        self.ic.awg.trigger()

    def handle_buffer(self, data, buffer_number=None):
        """
        See AcquisitionController
        :return:
        """
        self.buffer += data

    def post_acquire(self):
        """
        See AcquisitionController
        :return:
        """
        return self.buffer, self.buffers_per_acquisition, self.records_per_buffer, self.samples_per_record, self.time_array
        if self.number_of_channels == 2:
            # fit channel A and channel B
            #return [alazar.signal_to_volt(1, res1[0] + 127.5),
            #        alazar.signal_to_volt(2, res2[0] + 127.5),
            #        res1[1], res2[1],
            #        (res1[1] - res2[1]) % 360]
            return self.buffer, self.buffers_per_acquisition, self.records_per_buffer, self.samples_per_record, self.time_array
        else:
            raise Exception("Could not find CHANNEL_B during data extraction")
      
    def stop(self):
        self.ic.stop()

    def close(self):
        self.ic.atexit()
        self.alazar.close()
        super().close()
