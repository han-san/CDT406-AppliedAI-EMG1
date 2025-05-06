import ply.lex as lex
import ply.yacc as yacc

import experiment.exp_instructions as inst
import experiment.exp_states as exp_states

tokens = (
    "NUMBER",
    "TEXT",
    "WAIT",
    "LED_ON",
    "LED_OFF",
    "AUDIO_ON",
    "AUDIO_OFF",
    "LED",
    "DESCRIPTION",
    "PROTOCOL",
    "REST",
    "GRIP",
    "HOLD",
    "RELEASE",
    "FOR_BEGIN",
    "FOR_END",
    "FOR_LOOP"
)

t_ignore = " \t"

t_TEXT = r"\'[^\']*\'"
t_DESCRIPTION = r"Description:\s*\n"
t_PROTOCOL = r"Protocol:\s*\n"
t_FOR_LOOP = r"for"
t_FOR_BEGIN = r"begin"
t_FOR_END = r"end"


def t_NEWLINE(t):
    r"\n+"
    t.lexer.lineno += t.value.count("\n")


def t_WAIT(t):
    r"wait"
    t.type = "WAIT"
    return t


def t_LED_ON(t):
    r"led_on"
    t.type = "LED_ON"
    return t


def t_LED_OFF(t):
    r"led_off"
    t.type = "LED_OFF"
    return t

def t_AUDIO_ON(t):
    r"audio_on"
    t.type = "AUDIO_ON"
    return t

def t_AUDIO_OFF(t):
    r"audio_off"
    t.type = "AUDIO_OFF"
    return t

def t_NUMBER(t):
    r"[0-9]+(?:\.[0-9]+)?"
    t.value = float(t.value)
    return t

def t_LED(t):
	r"[RYG]"
	match t.value:
		case 'R':
			t.value = inst.ExperimentLEDColor.RED
		case 'Y':
			t.value = inst.ExperimentLEDColor.YELLOW
		case 'G':
			t.value = inst.ExperimentLEDColor.GREEN
	return t

def t_REST(t):
	r"rest:\s*\n"
	t.value = exp_states.ExperimentStates.REST.value
	return t

def t_GRIP(t):
	r"grip:\s*\n"
	t.value = exp_states.ExperimentStates.GRIP.value
	return t

def t_RELEASE(t):
	r"release:\s*\n"
	t.value = exp_states.ExperimentStates.RELEASE.value
	return t

def t_HOLD(t):
	r"hold:\s*\n"
	t.value = exp_states.ExperimentStates.HOLD.value
	return t

def t_error(t):
	print("Error parsing token " + str(t))
	exit(-1)

def p_program(p):
    """program : description timeline"""
    p[0] = (p[2], p[1])


def p_description(p):
    """description : DESCRIPTION TEXT"""
    p[0] = p[2]

def p_timeline(p):
    """timeline : PROTOCOL protocol"""
    p[0] = p[2]


def p_protocol(p):
    """protocol : state operands"""
    p[0] = [p[1]] + p[2]

def p_protocol_multiple(p):
    """protocol : state operands protocol"""
    p[0] = [p[1]] + p[2] + p[3]

def p_state(p):
    """state : REST
    | GRIP
    | HOLD
    | RELEASE"""
    p[0] = inst.ExperimentInstructionsChangeState(p[1])


def p_operands_single(p):
    """operands : operand"""
    p[0] = p[1]

def p_operands_multiple(p):
    """operands : operand operands"""
    p[0] = p[1] + p[2]

def p_operand(p):
    """operand : wait
    | control
    | for_loop"""
    p[0] = p[1]

def p_wait(p):
    """wait : WAIT NUMBER"""
    p[0] = [inst.ExperimentInstructionsWait(p[2])]

def p_control_led_on(p):
    """control : LED_ON LED"""
    p[0] = [inst.ExperimentInstructionsLEDOn(p[2])]

def p_control_led_off(p):
    """control : LED_OFF LED"""
    p[0] = [inst.ExperimentInstructionsLEDOff(p[2])]

def p_control_audio_on(p):
    """control : AUDIO_ON"""
    p[0] = [inst.ExperimentInstructionsAudioOn(0)]

def p_control_audio_off(p):
    """control : AUDIO_OFF"""
    p[0] = [inst.ExperimentInstructionsAudioOff(0)]

def p_for_loop_prot(p):
    """for_loop : FOR_LOOP NUMBER FOR_BEGIN protocol FOR_END"""
    operands = []
    for i in range(int(p[2])):
        operands += p[4]
    p[0] = operands

def p_error(p):
    print("Syntax error: ", p)


lexer = lex.lex()
parser = yacc.yacc()

def parse_script(dsl_code):
	return parser.parse(dsl_code)