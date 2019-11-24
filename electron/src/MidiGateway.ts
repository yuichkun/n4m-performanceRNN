export interface IMidiGateway {
  changeVelocity(value: number): void
  currentVelocity: number
  activeDevice: MidiDevice
}

interface MidiDevice {
  send(msg: MidiMessage, offset?: number): void
}

type MidiMessage = [number, number, number]

export class MidiGateway implements IMidiGateway {
  currentVelocity: number
  activeDevice: MidiDevice

  constructor() {
    this.currentVelocity = 100
    this.activeDevice = new StandardIo()
  }
  changeVelocity(value: number) {
    this.currentVelocity = value
  }
}

class StandardIo implements MidiDevice {
  // TODO: accept offset
  // @ts-ignore
  send(msg: MidiMessage, offset?: number) {
    msg.map(toString).forEach(m => {
      process.stdout.write(m)
    })
  }
}