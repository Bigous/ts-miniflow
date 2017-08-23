import { Input } from '../src/input';

test('Should greet with message', () => {
  const greeter = new Input('friend');
  expect(greeter.greet()).toBe('Bonjour, friend!');
});
