#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.1),
    on Fri Aug 30 10:47:59 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from load_stims
n_items = 665 
#items = [f'img/no_deck/object{i}.jpg' for i in range(1, n_items+1)]
#values = np.random.choice(value_set, len(n_items))
items = []
values = []

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.1'
expName = 'test'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1512, 982]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/qlu/Dropbox/github/emca-experiment/ds2016-psychopy/test-1col-pop_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('warning')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0.3255, 0.3255, 0.3255], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0.3255, 0.3255, 0.3255]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('advance_instruciton') is None:
        # initialise advance_instruciton
        advance_instruciton = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='advance_instruciton',
        )
    if deviceManager.getDevice('item_trial') is None:
        # initialise item_trial
        item_trial = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='item_trial',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "loader" ---
    
    # --- Initialize components for Routine "start" ---
    text_instruction = visual.TextStim(win=win, name='text_instruction',
        text='instruction \nbla bla bla ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    space_instruction_instruciton = visual.TextStim(win=win, name='space_instruction_instruciton',
        text='press [space] to continue',
        font='Arial',
        pos=(0, -.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    advance_instruciton = keyboard.Keyboard(deviceName='advance_instruciton')
    # Run 'Begin Experiment' code from code
    # init vars
    response = None
    reward = None 
    
    # --- Initialize components for Routine "context" ---
    # Run 'Begin Experiment' code from sample_trial_type_and_items
    # 1 based indexing 
    chosen_items = []
    chosen_values = []
    chosen_trial_ids = []
    trial_id = 0 
    text = visual.TextStim(win=win, name='text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "item_2afc" ---
    item_l = visual.ImageStim(
        win=win,
        name='item_l', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-.35, 0), draggable=False, size=(0.310, 0.363),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    item_r = visual.ImageStim(
        win=win,
        name='item_r', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0.35, 0), draggable=False, size=(0.310, 0.363),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=True, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    item_trial = keyboard.Keyboard(deviceName='item_trial')
    instruction_item_trial = visual.TextStim(win=win, name='instruction_item_trial',
        text='j or k',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    # Run 'Begin Experiment' code from process_feedback
    ## init vars
    #response = None
    #reward = None 
    show_current_trial_id = visual.TextStim(win=win, name='show_current_trial_id',
        text='',
        font='Arial',
        pos=(0, -.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    
    # --- Initialize components for Routine "feedback" ---
    feedback_response = visual.TextStim(win=win, name='feedback_response',
        text='',
        font='Arial',
        pos=(0, .3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    feedback_value = visual.TextStim(win=win, name='feedback_value',
        text='',
        font='Arial',
        pos=(0, -.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    show_trial_id_for_the_chosen_item = visual.TextStim(win=win, name='show_trial_id_for_the_chosen_item',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # set up handler to look after randomisation of conditions etc
    trials_2 = data.TrialHandler2(
        name='trials_2',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('trials-info-single-col-test.csv'), 
        seed=None, 
    )
    thisExp.addLoop(trials_2)  # add the loop to the experiment
    thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
    if thisTrial_2 != None:
        for paramName in thisTrial_2:
            globals()[paramName] = thisTrial_2[paramName]
    
    for thisTrial_2 in trials_2:
        currentLoop = trials_2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
        if thisTrial_2 != None:
            for paramName in thisTrial_2:
                globals()[paramName] = thisTrial_2[paramName]
        
        # --- Prepare to start Routine "loader" ---
        # create an object to store info about Routine loader
        loader = data.Routine(
            name='loader',
            components=[],
        )
        loader.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from load_stims
        items.append(image)
        values.append(value) 
        #print(len(values), value)
        # store start times for loader
        loader.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        loader.tStart = globalClock.getTime(format='float')
        loader.status = STARTED
        thisExp.addData('loader.started', loader.tStart)
        loader.maxDuration = None
        # keep track of which components have finished
        loaderComponents = loader.components
        for thisComponent in loader.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "loader" ---
        # if trial has changed, end Routine now
        if isinstance(trials_2, data.TrialHandler2) and thisTrial_2.thisN != trials_2.thisTrial.thisN:
            continueRoutine = False
        loader.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                loader.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in loader.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "loader" ---
        for thisComponent in loader.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for loader
        loader.tStop = globalClock.getTime(format='float')
        loader.tStopRefresh = tThisFlipGlobal
        thisExp.addData('loader.stopped', loader.tStop)
        # the Routine "loader" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 1.0 repeats of 'trials_2'
    
    
    # --- Prepare to start Routine "start" ---
    # create an object to store info about Routine start
    start = data.Routine(
        name='start',
        components=[text_instruction, space_instruction_instruciton, advance_instruciton],
    )
    start.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for advance_instruciton
    advance_instruciton.keys = []
    advance_instruciton.rt = []
    _advance_instruciton_allKeys = []
    # store start times for start
    start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    start.tStart = globalClock.getTime(format='float')
    start.status = STARTED
    thisExp.addData('start.started', start.tStart)
    start.maxDuration = None
    # keep track of which components have finished
    startComponents = start.components
    for thisComponent in start.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "start" ---
    start.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instruction* updates
        
        # if text_instruction is starting this frame...
        if text_instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instruction.frameNStart = frameN  # exact frame index
            text_instruction.tStart = t  # local t and not account for scr refresh
            text_instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instruction, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instruction.started')
            # update status
            text_instruction.status = STARTED
            text_instruction.setAutoDraw(True)
        
        # if text_instruction is active this frame...
        if text_instruction.status == STARTED:
            # update params
            pass
        
        # *space_instruction_instruciton* updates
        
        # if space_instruction_instruciton is starting this frame...
        if space_instruction_instruciton.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            space_instruction_instruciton.frameNStart = frameN  # exact frame index
            space_instruction_instruciton.tStart = t  # local t and not account for scr refresh
            space_instruction_instruciton.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(space_instruction_instruciton, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'space_instruction_instruciton.started')
            # update status
            space_instruction_instruciton.status = STARTED
            space_instruction_instruciton.setAutoDraw(True)
        
        # if space_instruction_instruciton is active this frame...
        if space_instruction_instruciton.status == STARTED:
            # update params
            pass
        
        # *advance_instruciton* updates
        waitOnFlip = False
        
        # if advance_instruciton is starting this frame...
        if advance_instruciton.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            advance_instruciton.frameNStart = frameN  # exact frame index
            advance_instruciton.tStart = t  # local t and not account for scr refresh
            advance_instruciton.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(advance_instruciton, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'advance_instruciton.started')
            # update status
            advance_instruciton.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(advance_instruciton.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(advance_instruciton.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if advance_instruciton.status == STARTED and not waitOnFlip:
            theseKeys = advance_instruciton.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _advance_instruciton_allKeys.extend(theseKeys)
            if len(_advance_instruciton_allKeys):
                advance_instruciton.keys = _advance_instruciton_allKeys[-1].name  # just the last key pressed
                advance_instruciton.rt = _advance_instruciton_allKeys[-1].rt
                advance_instruciton.duration = _advance_instruciton_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            start.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in start.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "start" ---
    for thisComponent in start.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for start
    start.tStop = globalClock.getTime(format='float')
    start.tStopRefresh = tThisFlipGlobal
    thisExp.addData('start.stopped', start.tStop)
    # check responses
    if advance_instruciton.keys in ['', [], None]:  # No response was made
        advance_instruciton.keys = None
    thisExp.addData('advance_instruciton.keys',advance_instruciton.keys)
    if advance_instruciton.keys != None:  # we had a response
        thisExp.addData('advance_instruciton.rt', advance_instruciton.rt)
        thisExp.addData('advance_instruciton.duration', advance_instruciton.duration)
    thisExp.nextEntry()
    # the Routine "start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=3.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "context" ---
        # create an object to store info about Routine context
        context = data.Routine(
            name='context',
            components=[text],
        )
        context.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from sample_trial_type_and_items
        # 0 = new new ; 1 = old new 
        u = randint(0, 100)
        if u > 60 or len(chosen_items) == 0: 
            trial_type = 0 
            trial_type_text = 'newnew -> encode'
        else:
            trial_type = 1 
            trial_type_text = 'oldnew -> retrieve'
        
        print(u, trial_type, trial_type_text)
        
        # sample two items and their values 
        if trial_type == 0:  # new new trial 
            image_l = items.pop()
            value_l = values.pop()
            image_r = items.pop()
            value_r = values.pop()
            chosen_trial_id_l = trial_id
            chosen_trial_id_r = trial_id
            is_old_l = 0 
            is_old_r = 0 
        elif trial_type == 1:  # old new trial 
            image_l = items.pop()
            value_l = values.pop()
            image_r = chosen_items.pop()
            value_r = chosen_values.pop()
            chosen_trial_id_l = trial_id
            chosen_trial_id_r = chosen_trial_ids.pop()
            is_old_l = 0 
            is_old_r = 1
        else:
            raise ValueError('invalid trial type', trial_type)
        
        
        if randint(0, 100) > 50: 
            image_l, image_r = image_r, image_l
            value_l, value_r = value_r, value_l
            is_old_l, is_old_r = is_old_r, is_old_l 
            chosen_trial_id_l, chosen_trial_id_r = chosen_trial_id_r, chosen_trial_id_l 
            
        
        text.setText(trial_type_text)
        # store start times for context
        context.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        context.tStart = globalClock.getTime(format='float')
        context.status = STARTED
        thisExp.addData('context.started', context.tStart)
        context.maxDuration = None
        # keep track of which components have finished
        contextComponents = context.components
        for thisComponent in context.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "context" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        context.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # if text is stopping this frame...
            if text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text.tStop = t  # not accounting for scr refresh
                    text.tStopRefresh = tThisFlipGlobal  # on global time
                    text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.stopped')
                    # update status
                    text.status = FINISHED
                    text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                context.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in context.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "context" ---
        for thisComponent in context.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for context
        context.tStop = globalClock.getTime(format='float')
        context.tStopRefresh = tThisFlipGlobal
        thisExp.addData('context.stopped', context.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if context.maxDurationReached:
            routineTimer.addTime(-context.maxDuration)
        elif context.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "item_2afc" ---
        # create an object to store info about Routine item_2afc
        item_2afc = data.Routine(
            name='item_2afc',
            components=[item_l, item_r, item_trial, instruction_item_trial, show_current_trial_id],
        )
        item_2afc.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        item_l.setImage(image_l)
        item_r.setImage(image_r)
        # create starting attributes for item_trial
        item_trial.keys = []
        item_trial.rt = []
        _item_trial_allKeys = []
        show_current_trial_id.setText(trial_id)
        # store start times for item_2afc
        item_2afc.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        item_2afc.tStart = globalClock.getTime(format='float')
        item_2afc.status = STARTED
        thisExp.addData('item_2afc.started', item_2afc.tStart)
        item_2afc.maxDuration = None
        # keep track of which components have finished
        item_2afcComponents = item_2afc.components
        for thisComponent in item_2afc.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "item_2afc" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        item_2afc.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *item_l* updates
            
            # if item_l is starting this frame...
            if item_l.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                item_l.frameNStart = frameN  # exact frame index
                item_l.tStart = t  # local t and not account for scr refresh
                item_l.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(item_l, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'item_l.started')
                # update status
                item_l.status = STARTED
                item_l.setAutoDraw(True)
            
            # if item_l is active this frame...
            if item_l.status == STARTED:
                # update params
                pass
            
            # *item_r* updates
            
            # if item_r is starting this frame...
            if item_r.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                item_r.frameNStart = frameN  # exact frame index
                item_r.tStart = t  # local t and not account for scr refresh
                item_r.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(item_r, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'item_r.started')
                # update status
                item_r.status = STARTED
                item_r.setAutoDraw(True)
            
            # if item_r is active this frame...
            if item_r.status == STARTED:
                # update params
                pass
            
            # *item_trial* updates
            waitOnFlip = False
            
            # if item_trial is starting this frame...
            if item_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                item_trial.frameNStart = frameN  # exact frame index
                item_trial.tStart = t  # local t and not account for scr refresh
                item_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(item_trial, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'item_trial.started')
                # update status
                item_trial.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(item_trial.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(item_trial.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if item_trial.status == STARTED and not waitOnFlip:
                theseKeys = item_trial.getKeys(keyList=['j','k'], ignoreKeys=["escape"], waitRelease=False)
                _item_trial_allKeys.extend(theseKeys)
                if len(_item_trial_allKeys):
                    item_trial.keys = _item_trial_allKeys[-1].name  # just the last key pressed
                    item_trial.rt = _item_trial_allKeys[-1].rt
                    item_trial.duration = _item_trial_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *instruction_item_trial* updates
            
            # if instruction_item_trial is starting this frame...
            if instruction_item_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                instruction_item_trial.frameNStart = frameN  # exact frame index
                instruction_item_trial.tStart = t  # local t and not account for scr refresh
                instruction_item_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instruction_item_trial, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instruction_item_trial.started')
                # update status
                instruction_item_trial.status = STARTED
                instruction_item_trial.setAutoDraw(True)
            
            # if instruction_item_trial is active this frame...
            if instruction_item_trial.status == STARTED:
                # update params
                pass
            
            # *show_current_trial_id* updates
            
            # if show_current_trial_id is starting this frame...
            if show_current_trial_id.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                show_current_trial_id.frameNStart = frameN  # exact frame index
                show_current_trial_id.tStart = t  # local t and not account for scr refresh
                show_current_trial_id.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(show_current_trial_id, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'show_current_trial_id.started')
                # update status
                show_current_trial_id.status = STARTED
                show_current_trial_id.setAutoDraw(True)
            
            # if show_current_trial_id is active this frame...
            if show_current_trial_id.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                item_2afc.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in item_2afc.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "item_2afc" ---
        for thisComponent in item_2afc.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for item_2afc
        item_2afc.tStop = globalClock.getTime(format='float')
        item_2afc.tStopRefresh = tThisFlipGlobal
        thisExp.addData('item_2afc.stopped', item_2afc.tStop)
        # check responses
        if item_trial.keys in ['', [], None]:  # No response was made
            item_trial.keys = None
        trials.addData('item_trial.keys',item_trial.keys)
        if item_trial.keys != None:  # we had a response
            trials.addData('item_trial.rt', item_trial.rt)
            trials.addData('item_trial.duration', item_trial.duration)
        # Run 'End Routine' code from process_feedback
        response = item_trial.keys 
        
        if response == 'j':
            chosen_item = image_l
            reward = value_l 
            is_old = is_old_l
            chosen_trial_id = chosen_trial_id_l
        elif response == 'k': 
            chosen_item = image_r
            reward = value_r 
            is_old = is_old_r
            chosen_trial_id = chosen_trial_id_r
        else:
            chosen_item = None 
            reward = None       
            is_old = None 
            chosen_trial_id = None 
        
        if trial_type == 0: # new new trial 
            chosen_items.append(chosen_item)
            chosen_values.append(reward) 
            chosen_trial_ids.append(trial_id)
        
        # decode the feedback / reward 
        feedback_img = f'img/feedback/blue/{reward}.jpg'
        
        # log data 
        thisExp.addData('response',  response)
        thisExp.addData('chosen_item', chosen_item)
        thisExp.addData('reward',  reward)
        thisExp.addData('is_old',  is_old)
        thisExp.addData('chosen_trial_id', chosen_trial_id)
        
        trial_id +=1 
        # the Routine "item_2afc" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "feedback" ---
        # create an object to store info about Routine feedback
        feedback = data.Routine(
            name='feedback',
            components=[feedback_response, feedback_value, show_trial_id_for_the_chosen_item],
        )
        feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        feedback_response.setText(response
        )
        feedback_value.setText(reward )
        show_trial_id_for_the_chosen_item.setText(chosen_trial_id)
        # store start times for feedback
        feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        feedback.tStart = globalClock.getTime(format='float')
        feedback.status = STARTED
        thisExp.addData('feedback.started', feedback.tStart)
        feedback.maxDuration = None
        # keep track of which components have finished
        feedbackComponents = feedback.components
        for thisComponent in feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "feedback" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedback_response* updates
            
            # if feedback_response is starting this frame...
            if feedback_response.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedback_response.frameNStart = frameN  # exact frame index
                feedback_response.tStart = t  # local t and not account for scr refresh
                feedback_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedback_response.started')
                # update status
                feedback_response.status = STARTED
                feedback_response.setAutoDraw(True)
            
            # if feedback_response is active this frame...
            if feedback_response.status == STARTED:
                # update params
                pass
            
            # if feedback_response is stopping this frame...
            if feedback_response.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedback_response.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    feedback_response.tStop = t  # not accounting for scr refresh
                    feedback_response.tStopRefresh = tThisFlipGlobal  # on global time
                    feedback_response.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback_response.stopped')
                    # update status
                    feedback_response.status = FINISHED
                    feedback_response.setAutoDraw(False)
            
            # *feedback_value* updates
            
            # if feedback_value is starting this frame...
            if feedback_value.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedback_value.frameNStart = frameN  # exact frame index
                feedback_value.tStart = t  # local t and not account for scr refresh
                feedback_value.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_value, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedback_value.started')
                # update status
                feedback_value.status = STARTED
                feedback_value.setAutoDraw(True)
            
            # if feedback_value is active this frame...
            if feedback_value.status == STARTED:
                # update params
                pass
            
            # if feedback_value is stopping this frame...
            if feedback_value.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedback_value.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    feedback_value.tStop = t  # not accounting for scr refresh
                    feedback_value.tStopRefresh = tThisFlipGlobal  # on global time
                    feedback_value.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback_value.stopped')
                    # update status
                    feedback_value.status = FINISHED
                    feedback_value.setAutoDraw(False)
            
            # *show_trial_id_for_the_chosen_item* updates
            
            # if show_trial_id_for_the_chosen_item is starting this frame...
            if show_trial_id_for_the_chosen_item.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                show_trial_id_for_the_chosen_item.frameNStart = frameN  # exact frame index
                show_trial_id_for_the_chosen_item.tStart = t  # local t and not account for scr refresh
                show_trial_id_for_the_chosen_item.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(show_trial_id_for_the_chosen_item, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'show_trial_id_for_the_chosen_item.started')
                # update status
                show_trial_id_for_the_chosen_item.status = STARTED
                show_trial_id_for_the_chosen_item.setAutoDraw(True)
            
            # if show_trial_id_for_the_chosen_item is active this frame...
            if show_trial_id_for_the_chosen_item.status == STARTED:
                # update params
                pass
            
            # if show_trial_id_for_the_chosen_item is stopping this frame...
            if show_trial_id_for_the_chosen_item.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > show_trial_id_for_the_chosen_item.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    show_trial_id_for_the_chosen_item.tStop = t  # not accounting for scr refresh
                    show_trial_id_for_the_chosen_item.tStopRefresh = tThisFlipGlobal  # on global time
                    show_trial_id_for_the_chosen_item.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'show_trial_id_for_the_chosen_item.stopped')
                    # update status
                    show_trial_id_for_the_chosen_item.status = FINISHED
                    show_trial_id_for_the_chosen_item.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback" ---
        for thisComponent in feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for feedback
        feedback.tStop = globalClock.getTime(format='float')
        feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('feedback.stopped', feedback.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if feedback.maxDurationReached:
            routineTimer.addTime(-feedback.maxDuration)
        elif feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        thisExp.nextEntry()
        
    # completed 3.0 repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
