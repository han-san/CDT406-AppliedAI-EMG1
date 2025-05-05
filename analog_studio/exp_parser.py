import ply.lex as lex
import ply.yacc as yacc

import exp_instructions as inst
import exp_states

tokens = (
    "NUMBER",
    "TEXT",
    "WAIT",
    "LED_ON",
    "LED_OFF",
    "LED",
    "DESCRIPTION",
    "PROTOCOL",
    "REST",
    "GRIP",
    "HOLD",
    "RELEASE",
)

t_ignore = " \t"

t_TEXT = r"\'[^\']*\'"
t_DESCRIPTION = r"Description:\s*\n"
t_PROTOCOL = r"Protocol:\s*\n"


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

script = []
comment = ''

def p_program(p):
    """program : description timeline"""
    pass


def p_description(p):
    """description : DESCRIPTION TEXT"""
    global comment
    comment = p[2]

def p_timeline(p):
    """timeline : PROTOCOL protocol"""
    pass


def p_protocol(p):
    """protocol : state operands
    | state operands protocol"""
    pass


def p_state(p):
    """state : REST
    | GRIP
    | HOLD
    | RELEASE"""
    script.append(inst.ExperimentInstructionsChangeState(p[1]))


def p_operands(p):
    """operands : operand
    | operand operands"""
    pass


def p_operand(p):
    """operand : wait
    | control"""
    pass


def p_wait(p):
    """wait : WAIT NUMBER"""
    script.append(inst.ExperimentInstructionsWait(p[2]))

def p_control_on(p):
    """control : LED_ON LED"""
    script.append(inst.ExperimentInstructionsLEDOn(p[2]))

def p_control_off(p):
    """control : LED_OFF LED"""
    script.append(inst.ExperimentInstructionsLEDOff(p[2]))

def p_error(p):
    print("Syntax error: ", p)


lexer = lex.lex()
parser = yacc.yacc()

def parse_script(dsl_code):
	global script
	script.clear()
	parser.parse(dsl_code)
	return (script,comment)