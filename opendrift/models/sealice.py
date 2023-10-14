# This file is part of OpenDrift.
#
# OpenDrift is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2
#
# OpenDrift is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenDrift.  If not, see <https://www.gnu.org/licenses/>.
#
# Copyright 2021, Julien Moreau, Plastic@Bay CIC

from datetime import timedelta, datetime
import numpy as np
import logging; logger = logging.getLogger(__name__)
#from opendrift.models.oceandrift import Lagrangian3DArray, OceanDrift
from opendrift.models.basemodel import OpenDriftSimulation
from opendrift.elements import LagrangianArray
from opendrift.models.physics_methods import hour_angle, solar_elevation

class SeaLiceElement(LagrangianArray):
    """
    Extending Lagrangian3DArray with specific properties for larval and
    juvenile stages of sea lice into super individuals
    """

    variables = LagrangianArray.add_variables([
        ('terminal_velocity', {'dtype': 'float',
                               'units': 'm/s',
                               'description': 'Terminal rise/sinking velocity (buoyancy) '
                               'in the ocean column',
                               'default': 0.}),
        ('current_drift_factor', {'dtype': 'float',
                                  'units': '1',
                                  'description': 'Elements are moved with this fraction of the '
                                  'current vector, in addition to currents '
                                  'and Stokes drift',
                                  'default': 1.}),                       
        ('LicePerFish',{'dtype':'float',
                        'units':'',
                        'description': '''Average number of lice attached to fishes''', 
                        'default':0.5}),
        ('AvFishW8',{'dtype':'float',
                    'units':'kg',
                    'description': '''Average weight of fishes''',
                    'default':4.5}),
        ('particle_biomass',{'dtype': 'float',
                            'units': 'kg',
                            'description': '''Fraction of farm biomass represented by particles''',
                            'default': 1000.}),
        ('hatched',{'dtype': 'float',
                            'units': '',
                            'description': '''Number of hatched nauplii in the superindividuals at spawn''', # this could be optimised
                            'default': 0.}),
        ('nauplii', {'dtype': 'float',
                            'units': '',
                            'description': '''Number of nauplii in the superindividual''',
                            'default': 0.}),
        ('copepodid', {'dtype': 'float',
                     'units': '',
                     'description': '''Number of copepodid in the superindividual''',
                     'default': 0.}),
        ('dead',  {'dtype': 'float',
                     'units': '',
                     'description': '''Number of dead lice in the superindividual''',
                     'default': 0.}),
        #('eliminated',{'dtype': 'int',
        #             'units': '',
        #             'description': '''Status of the particle when less than 1 individual''', # this should be optimised to bool or desactivated all together
        #             'default': 0}),
        ('degree_days', {'dtype': 'float',
                     'units': '',
                     'description': '''aging measure of superindividual (experimental)''',
                    'default': 0.}), #range 40-170
        ('safe_salinity_above', {'dtype': 'bool',
                     'units': '',
                     'description': '''Check if salinity allows vertical upward migration''', 
                    'default': False}),
        ('temperature_above', {'dtype': 'float',
                     'units': '',
                     'description': '''Temperature above the particles (depends of migration speed)''', 
                    'default': 10.}),
        ('temperature_below', {'dtype': 'float',
                     'units': '',
                     'description': '''Temperature below the particles (depends of migration speed)''', 
                    'default': 10.}),
        ('light', {'dtype': 'float',
                     'units': 'µmol photon s−1 m−2',
                     'description': '''Quantity of light received by the particles''', 
                     'default': 0.}),
                      ])


class SeaLice(OpenDriftSimulation):
    """
    Particle trajectory model based on the OpenDrift framework.
    Developed by Julien Moreau (Plastics@Bay CIC)
    Generic module for particles that are subject to vertical turbulent
    mixing with the possibility for positive or negative buoyancy
    Particles are sea-lice (Lepeophtheirus salmonis).
    """

    ElementType = SeaLiceElement

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
        'upward_sea_water_velocity': {'fallback': 0},
        # 'sea_surface_wave_significant_height': {'fallback': 0},
        # 'x_wind': {'fallback': 0},
        # 'y_wind': {'fallback': 0},
        'land_binary_mask': {'fallback': None},
        'sea_floor_depth_below_sea_level': {'fallback': 50},
        'surface_net_downward_radiative_flux':{'fallback': 0},
        'ocean_vertical_diffusivity': {'fallback': 0.01},
        'sea_water_temperature': {'fallback': 10},
        'sea_water_salinity': {'fallback': 34}
    }

    # required_profiles_z_range = [0, -50]  # The depth range (in m) which profiles should cover

    def __init__(self,*args, **kwargs):

        # Calling general constructor of parent class
        # super(SeaLice, self).__init__(*args, **kwargs)
        super().__init__(*args, **kwargs)

        # Configuration options
        self._add_config({
            'general:duration':{'type':'float', 'default':0.,
                                'min': None, 'max': None, 'units': 'seconds',
                                'description': 'Numerical experiment time in seconds',
                                'level': self.CONFIG_LEVEL_ESSENTIAL},
            'drift:vertical_advection': {'type': 'bool', 'default': True, 
                                'description':'Advect elements with vertical component of ocean current.',
                                'level': self.CONFIG_LEVEL_BASIC},
            'drift:vertical_mixing': {'type': 'bool', 'default': True, 
                                'level': self.CONFIG_LEVEL_BASIC,
                                'description': 'Activate vertical mixing scheme with inner loop'},
            'vertical_mixing:timestep': {'type': 'float', 'min': 0.1, 'max': 3600, 'default': 60,
                                'level': self.CONFIG_LEVEL_ADVANCED, 'units': 'seconds',
                                 'description':'Time step used for inner loop of vertical mixing.'},                    
            'lice:seeding_time_step':{'type':'float', 'default':None,
                                'min': None, 'max': None, 'units': 'seconds',
                                'description': 'Time between particle release',
                                'level': self.CONFIG_LEVEL_ESSENTIAL},
            'lice:death_rate':{'type':'float', 'default':0.01/3600,
                                'min': 0., 'max': None, 'units': 's-1',
                                'description': 'Rate of Larvae death per seconds',
                                'level': self.CONFIG_LEVEL_BASIC},
            'lice:maturation_rate':{'type':'float', 'default':0.1/3600,
                                'min': 0., 'max': None, 'units': 's-1',
                                'description': 'Rate of Nauplii maturation in Copepodids',
                                'level': self.CONFIG_LEVEL_BASIC},
            'lice:maturity_date':{'type':'float', 'default':3.63,
                                'min': 0., 'max': None, 'units': 'days',
                                'description': 'Days to start maturing into Copepodids',
                                'level': self.CONFIG_LEVEL_BASIC},
            'lice:sinking_velocity':{'type':'float', 'default':0.00025,
                                'min': 0., 'max': 0.01, 'units': 'm.s-1',
                                'description': 'Larvae sinking velocity',
                                'level': self.CONFIG_LEVEL_BASIC},
            'lice:vertical_migration_speed':{'type':'float', 'default':0.00075,
                                'min': 0., 'max': 0.01, 'units': 'm.s-1',
                                'description': 'Larvae vertical speed',
                                'level': self.CONFIG_LEVEL_BASIC},
            'lice:freezing_salinity':{'type':'float', 'default':27.,
                                'min': 0., 'max': 35., 'units': 'PSU',
                                'description': 'Salinity immobilising larvae',
                                'level': self.CONFIG_LEVEL_BASIC},
            'lice:avoided_salinity':{'type':'float', 'default':32.,
                                'min': 0., 'max': 50., 'units': 'PSU',
                                'description': 'Salinity actively avoided',
                                'level': self.CONFIG_LEVEL_BASIC},
            'lice:nu':{'type':'float', 'default':500.,
                                'min': 100, 'max': 1000, 'units': 'nm',
                                'description': 'Wavelength used to calculate irradiance',
                                'level': self.CONFIG_LEVEL_ADVANCED},
            'lice:k_water':{'type':'float', 'default':0.2,
                                'min': -10., 'max': 0., 'units': '',
                                'description': 'coefficient of exponential decay of light in water',
                                'level': self.CONFIG_LEVEL_ADVANCED},
            'lice:Nauplii_light_trigger':{'type':'float', 'default':2.E-5,
                                'min': 0., 'max': 1., 'units': 'µmol photon  s−1 m−2',
                                'description': 'light detection threshold of Nauplii',
                                'level': self.CONFIG_LEVEL_BASIC},
            'lice:Copepodid_light_trigger':{'type':'float', 'default':0.392,
                                'min': 0.0, 'max': 1.0, 'units': 'µmol photon  s−1 m−2',
                                'description': 'light detection threshold of copepodids',
                                'level': self.CONFIG_LEVEL_BASIC},
            'lice:twilight':{'type':'float', 'default':15,
                                'min': 0., 'max': 90., 'units': 'degrees',
                                'description': 'angle below the horizon for twilight',
                                'level': self.CONFIG_LEVEL_ADVANCED},
            'lice:sense_down':{'type':'bool', 'default':False,                                
                                'description': 'allow searching for warmer temp below (unlikely)',
                                'level': self.CONFIG_LEVEL_ADVANCED},                    
            })

    def prepare_run(self):
        logger.info("preparing the run...")
        ### need to find a better way of initialising the variables...
        self.prefix="lice:"
        self.vertical_migration_speed=self.get_config(self.prefix+'vertical_migration_speed') 
        self.sensing_distance=2*self.vertical_migration_speed
        self.tau= np.pi+2*np.deg2rad(self.get_config(self.prefix+'twilight')) #width of solar irridation distribution
        self.nu=self.get_config(self.prefix+'nu')
        self.sinking_velocity=self.get_config(self.prefix+'sinking_velocity')*-1
        self.freezing_salinity=self.get_config(self.prefix+'freezing_salinity')
        self.avoided_salinity=self.get_config(self.prefix+'avoided_salinity')
        self.k_water=self.get_config(self.prefix+'k_water')
        self.Nauplii_light_trigger=self.get_config(self.prefix+'Nauplii_light_trigger')
        self.Copepodid_light_trigger=self.get_config(self.prefix+'Copepodid_light_trigger')
        self.new_born()
        self.population()
        self.ref_date = datetime(1,1,1,0,0)

    def new_born(self):
        """
        Approach by Rittenhouse et al., (2016) [Rittenhouse, M.A., C.W. Revie
        and A. Hurford, 2016. A model for sea lice (Lepeophtheirus salmonis)
        dynamics in a seasonally changing environment. Energetics, 16, 8–16.,
        whereby the numbers of lice nauplii released every day (NP) are
        estimated according to the following equation:
        NP(t)=ηενA(t)
        where η = 592 is the number of eggs per clutch (Heuch et al., 2000;
        Rittenhouse et al., 2016),
        ε = 0.0476 is the egg string production rate per day (Heuch et al.,
        2000; Rittenhouse et al., 2016),
        ν = 0.6 is the hatching success i.e. the proportion of eggs which
        produce viable nauplii (Johnson and Albright, 1991),
        and A(t) is the total number of adult female lice on each farm,
        derived here from lice target levels and number of fish on site.
        """
        logger.debug("Calculating standard spawn of nauplii")
        Ne, nu = 592 , 0.6
        Eps=0.0476/timedelta(days=1).total_seconds()*self.get_config( \
                                                self.prefix+'seeding_time_step')
        self.spawn=Ne* Eps* nu

    def population(self):
        """
        Simulate the population evolution according to the biological parameters
        Parameters
        ----------
        t: numpy array (float)
        Time abscissa in seconds through the experiment
        self.Mat: int
        Age in timestep when the nauplii start to become adult copepodids
        maturation_rate: float
        Probability of maturation by time-step (t>= Mat)
        death_rate: float
        Probability of death by time-step

        Number of lice per particle when generated is standardised to 1
        Returns
        -------
        juv, adults, dead: np.arrays(int)
        Juveniles(Nauplii), adults(Copepodid), dead lices arrays through
        the duration of the experiment
        """
        logger.debug("Building global population model")
        death_rate=self.get_config(self.prefix+'death_rate')* self.time_step.total_seconds()
        maturation_rate=self.get_config(self.prefix+'maturation_rate')* self.time_step.total_seconds()
        duration = self.expected_steps_calculation #self.get_config('general:duration')/ self.time_step.total_seconds()
        Mat = int(np.ceil(self.get_config(self.prefix+'maturity_date')*24*3600/ \
                    self.time_step.total_seconds())) # maturity age in timestep
        t=np.arange(0,duration+1,dtype='int')
        self.juv=np.exp(-1*death_rate*t)
        # self.juv[0]=1
        self.dead=1-self.juv
        self.adult=np.zeros_like(t, dtype='float')
        if Mat<duration:
            decayjuv=self.juv[Mat]*np.exp(-1*(maturation_rate+death_rate)*(t[t>=Mat]-Mat))
            self.juv[t>=Mat] =decayjuv
            self.adult[t>=Mat]=1-self.dead[t>=Mat]-self.juv[t>=Mat]

    def SI_pop(self):
        """
        distribute the age fractions in the particles
        """
        logger.debug("Aging the super_individuals")
        scaling = self.elements.particle_biomass*self.elements.LicePerFish/ \
                    self.elements.AvFishW8
        #identify new spawns
        New_release = self.elements.age_seconds<=self.time_step.total_seconds()
        self.elements.hatched[New_release]=self.spawn*scaling[New_release]

        self.elements.nauplii[New_release]=self.elements.hatched[New_release]
        Free = ~New_release
        time_in_step=(self.elements.age_seconds[Free]/ \
                    self.time_step.total_seconds()).astype(np.int)
        self.elements.nauplii[Free]=self.elements.hatched[Free]* \
                                    self.juv[time_in_step]
        self.elements.copepodid[Free]=self.elements.hatched[Free]* \
                                    self.adult[time_in_step]
        self.elements.dead[Free]=self.elements.hatched[Free]* \
                                    self.dead[time_in_step]
        # desactivate dead particles
        ## this test is not good for seeds with few lice
        Dying=(self.elements.nauplii+ self.elements.copepodid)<1
        #self.elements.eliminated[Dying]=1
        self.deactivate_elements(Dying, reason="All individuals dead")


    def sensing(self):
        """
        Lice sensing if above or bellow the conditions are better for them
        within a specified distance.
        moving the particles up and then down then back in their initial position
        """
        logger.debug("sensing temperature and salinity")
        Backup= np.copy(self.elements.z[:])
        self.elements.z +=self.sensing_distance
        
        flying= self.elements.z>0
        nb_flying= flying.sum()
        if nb_flying>0:
            logger.debug(f"{nb_flying} elements sniffing the air, nose down")
            backup_flying=np.copy(self.elements.z[flying])
            self.elements.z[flying]=0
            
        self.elements.safe_salinity_above[self.environment.sea_water_salinity> \
                                            self.avoided_salinity]=True
        # print('safe sal ....', self.elements.safe_salinity_above)
        self.elements.temperature_above= self.environment.sea_water_temperature
        if nb_flying>0:
            self.elements.z[flying]=backup_flying
                    
        if self.get_config('lice:sense_down'):
            self.elements.z -= 2*self.sensing_distance
            Zmin=-1.*self.environment.sea_floor_depth_below_sea_level
            digging=self.elements.z<Zmin
            nb_digging=digging.sum()
            if nb_digging>0:
                logger.debug(f"{nb_digging} elements sniffing the ground, nose up")
                self.elements.z[digging]=Zmin
            self.elements.temperature_below= self.environment.sea_water_temperature
            self.elements.z = Backup
        else:
             self.elements.z -= self.sensing_distance

    def degree_days(self):
        """
        Calculate the degree days of a particles
        >>> under development <<<
        """
        logger.debug("accumulate degree_days **Experimental**")
        ### define active elements
        self.elements.degree_days+=self.environment.sea_water_temperature* \
                self.time_step.total_seconds()/timedelta(days=1).total_seconds()

    # def solar_noon(self):
    #     """
    #     Search solar noon at the longitude
    #     """
    #     self.ref_date=self.time.date.replace(hour=0,minute=0,second=0)
    #     Av_lon= np.mean(self.elements.lon)
    #     Av_lat= np.mean(self.elements.lat)
    #     minutes=np.arange(0,60*24, dtype='float')
    #     angles = np.empty_like(minutes)
    #     for i in range(len(minutes)):
    #         angles[i]=hour_angle(self.ref_date+timedelta(minutes=float(minutes[i])),Av_lon)
    #
    #     solar_noon=o.ref_date+timedelta(minutes=float(minutes[np.argmin(np.abs(angles))]))
    #     self.noon_elevation=solar_elevation(solar_noon, Av_lon, Av_lat)


    def irradiance(self):
        """
        Distribute the daily energy from irradiance with a gaussian distribution.
        We use the twilight times for high sensitivity organisms
        Convert irradiance from W.m-2 to micromol photon.s-1.m-2
        https://www.berthold.com/en/bioanalytic/knowledge/faq/irradiance-to-photon-flux
        calculate the photon flux in the water according to exponential decay from the sea surface
        """
        self.elements.solar_angle=np.deg2rad(hour_angle(self.time,np.mean(self.elements.lon)))
        solar_coeff= np.sqrt(self.tau/(2*np.pi)*np.exp(-self.tau/2*self.elements.solar_angle**2))
        self.elements.light = solar_coeff* \
                        self.environment.surface_net_downward_radiative_flux* \
                        self.nu*0.00836* \
                        np.exp(self.k_water*self.elements.z)

    def depth_test(self):
        """
        The natural range of the larvae is 0-50m
        """
        #self.elements.z[self.elements.z>=0]=0.0000001 #compensating for weird z=0 special case
        deepers=self.elements.z<-50
        nb_deepers=deepers.sum()
        if nb_deepers>0:
            logger.debug(f'{(self.elements.z<-50).sum()} elements lifted from -50m')
            self.elements.z[deepers]=-50

    def Lice_vertical_migration(self):
        """
        Make larval sea lice to migrate vertically according to salinity, light
        and temperature triggers.
        """
        ### all lice sink at the same speed
        #self.elements.terminal_velocityz = self.sinking_velocity

        ### filter actively avoiding salt
        # Frozen=self.environment.sea_water_salinity<self.freezing_salinity
        Avoiding = np.logical_and((self.freezing_salinity< \
                    self.environment.sea_water_salinity),
                    (self.environment.sea_water_salinity<self.avoided_salinity))

        Normal_salt = self.environment.sea_water_salinity>=self.avoided_salinity

        self.sensing()
        ### Irradiance computation
        self.irradiance()
        ### identify the elements involved in the different scenarios
        Filter_N= self.elements.copepodid < self.elements.nauplii
        Filter_C=~Filter_N
        safe_up_salt=Normal_salt&self.elements.safe_salinity_above
        #### need to be able to deal with false arrays of Filter_N and C...
        ### They generate empty arrays
        light_mig_N= safe_up_salt&Filter_N&(self.elements.light \
                    > self.Nauplii_light_trigger)

        light_mig_C= safe_up_salt&Filter_C&(self.elements.light \
                    > self.Copepodid_light_trigger)

        up_temp_mig= safe_up_salt&~light_mig_N&~light_mig_C& \
                    (self.elements.temperature_above \
                    > self.environment.sea_water_temperature)
        
        if self.get_config('lice:sense_down'):
            # conditions to swim down for temperature
            down_temp_mig=Normal_salt&~up_temp_mig&~light_mig_N& ~light_mig_C
            down_temp_mig =np.logical_and(down_temp_mig, self.elements.temperature_below> \
                        self.environment.sea_water_temperature)
            going_down=np.logical_or(Avoiding,down_temp_mig)
        else:
            going_down=Avoiding
        
        ### Active migration        
        going_up=np.logical_or(np.logical_or(light_mig_N,light_mig_C),up_temp_mig)
        logger.debug("{} going down, {} going up".format(going_down.sum(), going_up.sum()))
        self.elements.terminal_velocity=self.elements.moving*self.sinking_velocity
        self.elements.terminal_velocity[going_down] -=  self.vertical_migration_speed
        self.elements.terminal_velocity[going_up]   +=   self.vertical_migration_speed
        

    def vertical_motion(self, store_depths=False):
        """Mix particles vertically according to eddy diffusivity and buoyancy

            Buoyancy is expressed as terminal velocity, which is the
            steady-state vertical velocity due to positive or negative
            buoyant behaviour. It is usually a function of particle density,
            diameter, and shape.

            Vertical particle displacemend du to turbulent mixing is
            calculated using a random walk scheme" (Visser et al. 1996)
        """
        self.timer_start('main loop:updating elements:vertical motion')
        
        w = self.elements.terminal_velocity
        #vertical advection component
        if self.get_config('drift:vertical_advection'):
            w+=self.environment.upward_sea_water_velocity
        else:
            logger.debug('Vertical advection deactivated')
                   
        dt_mix = self.get_config('vertical_mixing:timestep')
        ntimes_mix = np.abs(int(self.time_step.total_seconds()/dt_mix))
        Kz=self.fallback_values['ocean_vertical_diffusivity']
        Zmin = -1.*self.environment.sea_floor_depth_below_sea_level
        logger.debug('Using constant diffusivity specified by fallback_values[''ocean_vertical_diffusivity''] = %s m2.s-1' % (self.fallback_values['ocean_vertical_diffusivity']))
        for i in range(0, ntimes_mix):           
            # New position  =  old position   + dz turbulence   + dz terminal velocity and advection (optional)
           
            self.elements.z = self.elements.z + \
                              self.elements.moving*( \
                                   np.sqrt(2*Kz)*np.random.normal (0, dt_mix**.5, (self.num_elements_active())) \
                                   + w*dt_mix \
                              )
                
            # Reflect from surface
            reflect = self.elements.z > 0
            if reflect.sum() > 0:
                self.elements.z[reflect] = -self.elements.z[reflect]

            # Reflect elements going below seafloor
            #bottom = np.where(np.logical_and(self.elements.z < Zmin, self.elements.moving == 1))
            bottom=self.elements.z< Zmin
            nb_bottom=bottom.sum()
            if nb_bottom > 0:
                logger.debug('%s elements penetrated seafloor, lifting up' % nb_bottom)
                self.elements.z[bottom] = 2*Zmin[bottom] - self.elements.z[bottom]
                # need a special case when very shallow as particles can ping-pong
                # will put it at the sea surface
                pingpong=self.elements.z[bottom]>0
                nb_pingpong=pingpong.sum()
                if nb_pingpong >0:
                    # there needs a configuration option to decide what to do in case of ping pong
                    # For the lice the surface is ok as constant K but seabed could also be a solution or right in the middle
                    logger.debug('%s elements penetrated seafloor then breached surface => in the middle' % nb_pingpong)
                    self.elements.z[pingpong] =Zmin[pingpong]/2

        self.timer_end('main loop:updating elements:vertical motion')   
    
        
    def update(self):
        self.SI_pop()
        #self.degree_days()
        self.advect_ocean_current()        
        self.Lice_vertical_migration()
                
        # Turbulent Mixing
        if self.get_config('drift:vertical_mixing') is True:
            self.vertical_motion()
        else:
            # to be implemented
            self.vertical_buoyancy()
        self.depth_test()
        
